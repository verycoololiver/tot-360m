"""Unit test verifying train_tpu_calibration.py accounting, checkpointing, and cursor logic."""

import os
import sys
import time
import json
import shutil
import tempfile
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tot_360m.config import TotMiniConfig
from tot_360m.src.model import TotForCausalLM
from tot_360m.src.checkpoint import save_checkpoint, load_checkpoint, get_config_fingerprint
from tot_360m.src.train_tpu_calibration import (
    PackedBinaryDataset,
    generate_synthetic_shards_for_testing,
    train_worker,
)


class DummyArgs:
    def __init__(self, data_dir, output_dir, resume_ckpt=None):
        self.config = "mini"
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.resume_ckpt = resume_ckpt
        self.allow_synthetic_shards = True
        self.max_tokens = 2000
        self.max_seconds = 300
        self.seq_len = 32
        self.batch_size = 2
        self.grad_accum_steps = 1
        self.lr = 1e-3
        self.max_grad_norm = 1.0
        self.seed = 20261009
        self.log_interval = 1
        self.save_interval = 5
        self.keep_checkpoints = 2
        self.precision = "fp32"


class TestTPURunnerLogic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.output_dir = os.path.join(self.temp_dir, "checkpoints")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_data_raises_without_explicit_flag(self):
        """Verify that missing real shards raises FileNotFoundError when synthetic flag is False."""
        args = DummyArgs(self.data_dir, self.output_dir)
        args.allow_synthetic_shards = False
        with self.assertRaises(FileNotFoundError):
            train_worker(0, args, t_session_start=time.time())

    def test_missing_resume_checkpoint_raises(self):
        """Verify that a nonexistent resume checkpoint path explicitly raises FileNotFoundError."""
        args = DummyArgs(self.data_dir, self.output_dir, resume_ckpt="non_existent_path_to_ckpt.pt")
        with self.assertRaises(FileNotFoundError):
            train_worker(0, args, t_session_start=time.time())

    def test_compute_all_rank_cursors_multi_shard(self):
        """Verify deterministic per-rank cursor calculation across multiple shards and 8 ranks."""
        from tot_360m.src.train_tpu_calibration import compute_all_rank_cursors
        # 16 shards, 50 sequences each (seq_len=32, int32 = 128 bytes per seq -> 6400 bytes per shard)
        os.makedirs(self.data_dir, exist_ok=True)
        shards = []
        for i in range(16):
            s_path = os.path.join(self.data_dir, f"test_shard_{i:02d}.bin")
            with open(s_path, "wb") as f:
                f.write(b"\x00" * (50 * 32 * 4))
            shards.append(s_path)

        # 8 ranks, 2 shards per rank (100 seqs per rank total).
        # Consume 75 sequences per rank -> should point to shard_idx 1, offset 25
        cursors = compute_all_rank_cursors(
            all_shards=shards,
            world_size=8,
            seq_len=32,
            seqs_consumed_per_rank=75,
            dtype=torch.int32,
        )
        self.assertEqual(len(cursors), 8)
        for r in range(8):
            self.assertEqual(cursors[str(r)]["shard_idx"], 1)
            self.assertEqual(cursors[str(r)]["seq_offset"], 25)

    def test_compute_rank_remaining_sequences_multi_shard(self):
        """Verify exact remaining sequence calculation subtracting consumed preceding shards."""
        from tot_360m.src.train_tpu_calibration import compute_rank_remaining_sequences
        os.makedirs(self.data_dir, exist_ok=True)
        # Create 2 shards with 50 sequences each
        shards = []
        for i in range(2):
            s_path = os.path.join(self.data_dir, f"rem_shard_{i:02d}.bin")
            with open(s_path, "wb") as f:
                f.write(b"\x00" * (50 * 32 * 4))
            shards.append(s_path)

        # Cursor at (1, 25): shard 0 is fully consumed (50 seqs), shard 1 has 25 consumed -> exactly 25 remain
        rem = compute_rank_remaining_sequences(shards, shard_idx=1, seq_offset=25, seq_len=32, dtype=torch.int32)
        self.assertEqual(rem, 25, f"Expected 25 remaining sequences, got {rem}")

        # Cursor at (0, 25): shard 0 has 25 consumed, shard 1 is unconsumed -> exactly 75 remain
        rem0 = compute_rank_remaining_sequences(shards, shard_idx=0, seq_offset=25, seq_len=32, dtype=torch.int32)
        self.assertEqual(rem0, 75, f"Expected 75 remaining sequences, got {rem0}")

        # Cursor at (2, 0): all shards exhausted -> exactly 0 remain
        rem_end = compute_rank_remaining_sequences(shards, shard_idx=2, seq_offset=0, seq_len=32, dtype=torch.int32)
        self.assertEqual(rem_end, 0, f"Expected 0 remaining sequences, got {rem_end}")

    def test_save_resume_cursor_continuation(self):
        """Verify initial run, checkpoint save, and resumed run cursor continuity."""
        # 1. Run initial 10 steps and save checkpoint
        args1 = DummyArgs(self.data_dir, self.output_dir)
        args1.max_tokens = 640  # 10 steps * 62 targets
        args1.save_interval = 5
        train_worker(0, args1, t_session_start=time.time())

        ckpt_step_5 = os.path.join(self.output_dir, "checkpoint_step_00005.pt")
        self.assertTrue(os.path.exists(ckpt_step_5), f"Checkpoint not found: {ckpt_step_5}")

        # Check saved checkpoint data_state has rank_cursors
        ckpt = torch.load(ckpt_step_5, map_location="cpu", weights_only=False)
        self.assertIn("rank_cursors", ckpt["data_state"])
        self.assertIn("0", ckpt["data_state"]["rank_cursors"])

        # Check initial run calibration summary
        summary1_path = os.path.join(self.output_dir, "calibration_summary.json")
        with open(summary1_path, "r", encoding="utf-8") as f:
            s1 = json.load(f)
        # Verify strict cap compliance: 640 cap with 62 targets/step must stop at 620 (10 steps)
        self.assertLessEqual(s1["total_valid_targets"], 640, f"Cap exceeded: {s1['total_valid_targets']} > 640")
        self.assertEqual(s1["total_valid_targets"], 620, f"Expected exactly 620 targets (10 steps), got {s1['total_valid_targets']}")
        self.assertEqual(s1["tokens_trained_this_run"], 620)
        self.assertGreater(s1["final_loss"], 0.0, "final_loss must report last valid step loss, not zero")

        # 2. Resume from step 5
        args2 = DummyArgs(self.data_dir, self.output_dir, resume_ckpt=ckpt_step_5)
        args2.max_tokens = 960  # continue for more tokens
        train_worker(0, args2, t_session_start=time.time())

        final_summary = os.path.join(self.output_dir, "calibration_summary.json")
        self.assertTrue(os.path.exists(final_summary))
        with open(final_summary, "r", encoding="utf-8") as f:
            s2 = json.load(f)
        self.assertLessEqual(s2["total_valid_targets"], 960)
        self.assertEqual(s2["total_valid_targets"], 930, f"Expected 930 targets, got {s2['total_valid_targets']}")
        self.assertEqual(s2["tokens_trained_this_run"], 620)  # 930 - 310 = 620 targets trained this session
        self.assertGreater(s2["final_loss"], 0.0)

    def test_zero_update_when_budget_already_exhausted(self):
        """Verify that when tokens_seen already equals or exceeds max_tokens, 0 updates occur."""
        # 1. Run to 620 targets
        args1 = DummyArgs(self.data_dir, self.output_dir)
        args1.max_tokens = 640
        train_worker(0, args1, t_session_start=time.time())

        final_ckpt = os.path.join(self.output_dir, "tot_360m_calibration_final.pt")
        self.assertTrue(os.path.exists(final_ckpt))

        # 2. Attempt to resume with max_tokens = 620 (already satisfied)
        args2 = DummyArgs(self.data_dir, self.output_dir, resume_ckpt=final_ckpt)
        args2.max_tokens = 620
        train_worker(0, args2, t_session_start=time.time())

        summary_path = os.path.join(self.output_dir, "calibration_summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        self.assertEqual(s["tokens_trained_this_run"], 0, "Expected 0 updates when budget already satisfied")
        self.assertEqual(s["total_valid_targets"], 620)


    def test_packed_binary_dataset_uint16_and_int32(self):
        """Verify exact known token sequences for both uint16 and int32 storage formats."""
        # 1. Test uint16
        seq_len = 6
        uint16_tokens = torch.tensor([
            [0, 1, 42, 500, 31999, 65535],
            [10, 20, 30, 40, 50, 60],
            [31995, 31996, 31997, 31998, 31999, 0],
        ], dtype=torch.int64)

        u16_path = os.path.join(self.data_dir, "test_u16.bin")
        os.makedirs(self.data_dir, exist_ok=True)
        with open(u16_path, "wb") as f:
            f.write(uint16_tokens.to(torch.uint16).numpy().tobytes())

        ds_u16 = PackedBinaryDataset([u16_path], seq_len=seq_len, dtype=torch.uint16)
        self.assertEqual(ds_u16.total_seqs, 3)
        read_u16 = [tensor for _, _, tensor in ds_u16.get_sequence_iterator()]
        self.assertEqual(len(read_u16), 3)
        for expected, actual in zip(uint16_tokens, read_u16):
            self.assertTrue(torch.equal(expected, actual), f"uint16 mismatch: expected {expected}, got {actual}")

        # 2. Test int32
        int32_tokens = torch.tensor([
            [0, 1, 42, 500, 31999, 1000000],
            [100, 200, 300, 400, 500, 600],
            [200000, 300000, 400000, 500000, 600000, 700000],
        ], dtype=torch.int64)

        i32_path = os.path.join(self.data_dir, "test_i32.bin")
        with open(i32_path, "wb") as f:
            f.write(int32_tokens.to(torch.int32).numpy().tobytes())

        ds_i32 = PackedBinaryDataset([i32_path], seq_len=seq_len, dtype=torch.int32)
        self.assertEqual(ds_i32.total_seqs, 3)
        read_i32 = [tensor for _, _, tensor in ds_i32.get_sequence_iterator()]
        self.assertEqual(len(read_i32), 3)
        for expected, actual in zip(int32_tokens, read_i32):
            self.assertTrue(torch.equal(expected, actual), f"int32 mismatch: expected {expected}, got {actual}")

    def test_reset_data_cursor_and_batch_size_change_preserves_cursor_and_expands_tmax(self):
        """Verify --reset-data-cursor with changed batch_size computes exact new cursor and expands scheduler T_max."""
        # Phase 1: batch_size=1, grad_accum_steps=1, 10 steps * 31 targets = 310 targets
        args1 = DummyArgs(self.data_dir, self.output_dir)
        args1.batch_size = 1
        args1.grad_accum_steps = 1
        args1.max_tokens = 310
        args1.save_interval = 10
        train_worker(0, args1, t_session_start=time.time())

        phase1_ckpt = os.path.join(self.output_dir, "tot_360m_calibration_final.pt")
        self.assertTrue(os.path.exists(phase1_ckpt))

        # Phase 2: resume from step 10 with batch_size=2 (62 targets/step), --reset-data-cursor, and larger max_tokens
        out_dir2 = os.path.join(self.temp_dir, "checkpoints_phase2")
        args2 = DummyArgs(self.data_dir, out_dir2, resume_ckpt=phase1_ckpt)
        args2.batch_size = 2
        args2.grad_accum_steps = 1
        args2.max_tokens = 620  # 310 remaining -> exactly 5 steps at 62 targets/step (steps 10..15)
        args2.save_interval = 5
        args2.skip_identity_verify = True
        args2.reset_data_cursor = True
        train_worker(0, args2, t_session_start=time.time())

        phase2_ckpt = os.path.join(out_dir2, "tot_360m_calibration_final.pt")
        self.assertTrue(os.path.exists(phase2_ckpt))
        ckpt2 = torch.load(phase2_ckpt, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt2["step"], 15)
        self.assertEqual(ckpt2["data_state"]["tokens_seen"], 620)
        # Only 5 steps * 2 seqs = 10 sequences were consumed in Phase 2 after cursor reset (NOT 15 * 2 = 30!)
        self.assertEqual(ckpt2["data_state"]["rank_cursors"]["0"]["shard_idx"], 0)
        self.assertEqual(ckpt2["data_state"]["rank_cursors"]["0"]["seq_offset"], 10)
        self.assertEqual(ckpt2["data_state"]["global_seqs_consumed_in_pack"], 10)


if __name__ == "__main__":
    unittest.main()
