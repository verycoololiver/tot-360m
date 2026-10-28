"""Local miniature GPU-candidate tests; these do not prove actual CUDA speed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from tot_360m.config import Tot360MConfig
from tot_360m.src.model import TotForCausalLM
from tot_360m.src.train_gpu_calibration import (
    GlobalCursor, admit_device, load_payload, recursive_equal, validate_manifest,
)


def make_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"vocab_size": 1000, "seq_len": 16, "dtype": "uint16", "tokenizer_json_sha256": "local-fixture", "shards": []}
    # Uneven shards exercise boundary crossing and final incomplete groups.
    for i, count in enumerate([3, 7, 9]):
        path = root / f"shard_{i:03d}.bin"
        ids = (np.arange(count * 16, dtype=np.uint16) + i * 31) % 1000
        buf = ids.astype("<u2").tobytes()
        path.write_bytes(buf)
        manifest["shards"].append({"name": path.name, "seq_count": count, "sha256": hashlib.sha256(buf).hexdigest()})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class GPUCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tot_gpu_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fail_closed_hardware_and_precision(self):
        for args in [("baseline", "cpu", False, "fp32", False, 0, 1, False),
                     ("baseline", "cuda", False, "fp16", False, 0, 1, False),
                     ("baseline", "cuda", False, "bf16", True, 2, 2, False),
                     ("baseline", "cuda", True, "fp16", True, 2, 2, False)]:
            with self.assertRaises(RuntimeError):
                admit_device(*args)
        admit_device("baseline", "cuda", False, "fp16", True, 2, 2, False)

    def test_full_baseline_parameter_count_on_meta(self):
        with torch.device("meta"):
            model = TotForCausalLM(Tot360MConfig())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 345359296)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_disjoint_rank_plans_and_no_dropped_tail(self):
        shards = [{"name": "a", "count": 3}, {"name": "b", "count": 8}, {"name": "c", "count": 8}]
        cursor = GlobalCursor(shards, 16, position=1)
        ids = cursor.plan(8)
        rank0 = ids[0:2] + ids[4:6]
        rank1 = ids[2:4] + ids[6:8]
        self.assertEqual(set(rank0) & set(rank1), set())
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(cursor.position, 1)  # planning/failed updates never commit
        cursor.position += 8
        self.assertEqual(len(cursor.plan(8)), 8)
        cursor.position += 8
        self.assertEqual(cursor.plan(2), [("c", 6), ("c", 7)])
        with self.assertRaises(ValueError):
            cursor.plan(3)

    def test_corrupt_shards_fail_before_model_allocation(self):
        from tot_360m.config import TotMiniConfig
        root = make_fixture(self.root / "data")
        validate_manifest(root, TotMiniConfig(), 16, fixture=True)
        path = root / "shard_000.bin"
        path.write_bytes(path.read_bytes() + b"\x01")
        with self.assertRaises(ValueError):
            validate_manifest(root, TotMiniConfig(), 16, fixture=True)

    def invoke(self, data, output, max_steps, resume=None, precision="fp32", ddp=False, max_tokens=1000):
        args = ["-m", "tot_360m.src.train_gpu_calibration", "--config", "mini", "--device", "cpu", "--fixture",
                "--precision", precision, "--data-dir", str(data), "--output-dir", str(output),
                "--seq-len", "16", "--batch-size", "1" if ddp else "2", "--grad-accum-steps", "1",
                "--max-tokens", str(max_tokens), "--max-steps", str(max_steps), "--schedule-steps", "20",
                "--max-seconds", "180", "--save-margin", "5", "--save-interval", "1", "--warmup-steps", "1",
                "--record-batch-ids"]
        if resume:
            args += ["--resume", str(resume)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        env["OMP_NUM_THREADS"] = "2"
        env["PYTHONIOENCODING"] = "utf-8"
        env["USE_LIBUV"] = "0"  # Windows CPU wheel lacks libuv; Linux CUDA probe is separate.
        if ddp:
            # Windows's wheel cannot launch TCPStore's libuv path. Explicit
            # CPU-only FileStore tests DDP logic without changing the CUDA route.
            args += ["--fixture-rendezvous", str(output.parent / (output.name + ".rdzv"))]
            workers = []
            for rank in (0, 1):
                worker_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
                workers.append(subprocess.Popen([sys.executable] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env))
            try:
                for p in workers:
                    stdout, stderr = p.communicate(timeout=240)
                    self.assertEqual(p.returncode, 0, stdout[-3000:] + stderr[-5000:])
            finally:
                for p in workers:
                    if p.poll() is None:
                        p.kill()
                        p.communicate(timeout=10)
        else:
            p = subprocess.run([sys.executable] + args, capture_output=True, text=True, env=env, timeout=240)
            self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-5000:])
        return json.loads((output / "summary.json").read_text()), torch.load(output / "final.pt", weights_only=False, map_location="cpu")

    def test_fresh_process_fp16_scaler_optimizer_rng_cursor_recovery(self):
        data = make_fixture(self.root / "data")
        ctrl, control = self.invoke(data, self.root / "control", 3, precision="fp16")
        first, checkpoint = self.invoke(data, self.root / "first", 1, precision="fp16")
        resumed, final = self.invoke(data, self.root / "resumed", 2, resume=self.root / "first/final.pt", precision="fp16")
        self.assertEqual(control["tokens_seen"], 90)
        self.assertEqual(final["step"], 3)
        self.assertEqual(resumed["targets_this_run"], 60)
        self.assertTrue(checkpoint["scaler_state"])
        for key in ("model_state", "optimizer_state", "scheduler_state", "scaler_state", "rank_states", "cursor", "tokens_seen", "step"):
            recursive_equal(control[key], final[key])
        self.assertEqual(ctrl["batch_ids"], first["batch_ids"] + resumed["batch_ids"])
        self.assertTrue(any(not torch.equal(checkpoint["model_state"][k], final["model_state"][k]) for k in final["model_state"]))
        del checkpoint["scaler_state"]
        broken = self.root / "missing_scaler.pt"
        torch.save(checkpoint, broken)
        with self.assertRaises(ValueError):
            load_payload(broken, control["identity"], require_scaler=True)

    def test_cap_below_next_step_preserves_cursor(self):
        data = make_fixture(self.root / "data")
        report, state = self.invoke(data, self.root / "cap", 3, max_tokens=29)
        self.assertEqual(state["step"], 0)
        self.assertEqual(state["cursor"], 0)
        self.assertEqual(report["total_valid_targets"], 0)
        self.assertIsNone(report["steady_state_targets_per_second"])

    def test_two_rank_fresh_process_recovery(self):
        data = make_fixture(self.root / "ddp_data")
        ctrl, control = self.invoke(data, self.root / "ddp_control", 3, ddp=True)
        first, _ = self.invoke(data, self.root / "ddp_first", 1, ddp=True)
        resumed, final = self.invoke(data, self.root / "ddp_resumed", 2, ddp=True, resume=self.root / "ddp_first/final.pt")
        self.assertEqual(final["tokens_seen"], 90)
        self.assertEqual(len(final["rank_states"]), 2)
        for key in ("model_state", "optimizer_state", "scheduler_state", "rank_states", "cursor", "tokens_seen", "step"):
            recursive_equal(control[key], final[key], atol=1e-7)
        self.assertEqual(ctrl["batch_ids"], first["batch_ids"] + resumed["batch_ids"])

    def test_cross_backend_tpu_to_gpu_and_gpu_to_tpu_relay(self):
        from tot_360m.config import TotMiniConfig
        from tot_360m.src.checkpoint import save_checkpoint, load_checkpoint
        data = make_fixture(self.root / "relay_data")
        cfg = TotMiniConfig()
        model = TotForCausalLM(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=3e-5)
        tpu_ckpt = self.root / "tpu_step_2.pt"
        save_checkpoint(
            str(tpu_ckpt),
            step=2,
            model=model,
            optimizer=opt,
            scheduler=sched,
            config=cfg,
            data_state={"tokens_seen": 60, "global_seqs_consumed_in_pack": 4, "rank_cursors": {"0": {"shard_idx": 1, "seq_offset": 1}}},
        )
        # Resume GPU runner from TPU v2 checkpoint with --allow-cross-backend
        gpu_out = self.root / "gpu_from_tpu"
        args = [
            sys.executable, "-m", "tot_360m.src.train_gpu_calibration",
            "--config", "mini", "--device", "cpu", "--fixture", "--precision", "fp32",
            "--data-dir", str(data), "--output-dir", str(gpu_out),
            "--seq-len", "16", "--batch-size", "2", "--grad-accum-steps", "1",
            "--max-tokens", "1000", "--max-steps", "1", "--schedule-steps", "20",
            "--max-seconds", "180", "--save-margin", "5", "--save-interval", "1", "--warmup-steps", "1",
            "--resume", str(tpu_ckpt), "--allow-cross-backend",
        ]
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]), OMP_NUM_THREADS="2")
        p = subprocess.run(args, capture_output=True, text=True, env=env, timeout=180)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        gpu_final = torch.load(gpu_out / "final.pt", map_location="cpu", weights_only=False)
        self.assertEqual(gpu_final["step"], 3)
        self.assertEqual(gpu_final["tokens_seen"], 90)
        self.assertEqual(gpu_final["cursor"], 6)
        self.assertEqual(gpu_final["scheduler_state"]["T_max"], 20)

        # Now load the GPU checkpoint back into TPU load_checkpoint
        model2 = TotForCausalLM(cfg)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=20, eta_min=3e-5)
        step2, data_state2, _ = load_checkpoint(str(gpu_out / "final.pt"), model2, opt2, sched2, cfg, expected_identity=None)
        self.assertEqual(step2, 3)
        self.assertEqual(data_state2["tokens_seen"], 90)
        self.assertEqual(data_state2["global_seqs_consumed_in_pack"], 6)

    def test_eight_rank_tpu_midpack_handoff_no_duplicate_sequences_and_schedule_auto_expand(self):
        from tot_360m.config import TotMiniConfig
        from tot_360m.src.checkpoint import save_checkpoint
        # Create 8 shards (1 per TPU rank), each with 4 sequences of length 16 (total 32 sequences)
        data = self.root / "eight_rank_data"
        data.mkdir(parents=True, exist_ok=True)
        manifest = {"vocab_size": 1000, "seq_len": 16, "dtype": "uint16", "tokenizer_json_sha256": "local-fixture", "shards": []}
        for i in range(8):
            path = data / f"shard_{i:03d}.bin"
            ids = (np.arange(4 * 16, dtype=np.uint16) + i * 17) % 1000
            buf = ids.astype("<u2").tobytes()
            path.write_bytes(buf)
            manifest["shards"].append({"name": path.name, "seq_count": 4, "sha256": hashlib.sha256(buf).hexdigest()})
        (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        # Simulate 8 TPU ranks having consumed 2 sequences from each of the 8 shards (16 sequences total) at step=50
        tpu_consumed = {(f"shard_{r:03d}.bin", off) for r in range(8) for off in (0, 1)}
        rank_cursors = {str(r): {"shard_idx": 0, "seq_offset": 2} for r in range(8)}
        cfg = TotMiniConfig()
        model = TotForCausalLM(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=3e-5)
        tpu_ckpt = self.root / "tpu_8rank_step50.pt"
        save_checkpoint(
            str(tpu_ckpt),
            step=50,
            model=model,
            optimizer=opt,
            scheduler=sched,
            config=cfg,
            data_state={
                "tokens_seen": 50 * 16 * 15,
                "global_seqs_consumed_in_pack": 16,
                "rank_cursors": rank_cursors,
            },
        )
        # Deliberately pass --schedule-steps 10 (< step 50) to verify auto-expansion on cross-backend bulk resume
        gpu_out1 = self.root / "gpu_hop1"
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]), OMP_NUM_THREADS="2")
        p1 = subprocess.run([
            sys.executable, "-m", "tot_360m.src.train_gpu_calibration",
            "--config", "mini", "--device", "cpu", "--fixture", "--precision", "fp32",
            "--data-dir", str(data), "--output-dir", str(gpu_out1),
            "--seq-len", "16", "--batch-size", "4", "--grad-accum-steps", "1",
            "--max-tokens", "50000", "--max-steps", "2", "--schedule-steps", "10",
            "--max-seconds", "180", "--save-margin", "5", "--save-interval", "1", "--warmup-steps", "1",
            "--record-batch-ids", "--resume", str(tpu_ckpt),
            "--allow-cross-backend", "--allow-bulk-manifest", "--skip-checksum-verify",
        ], capture_output=True, text=True, env=env, timeout=180)
        self.assertEqual(p1.returncode, 0, p1.stdout + p1.stderr)
        sum1 = json.loads((gpu_out1 / "summary.json").read_text())
        self.assertEqual(sum1["steps_this_run"], 2)
        self.assertEqual(sum1["global_step"], 52)
        hop1_ids = [tuple(x) for x in sum1["batch_ids"]]
        self.assertEqual(len(hop1_ids), 8)
        self.assertEqual(set(hop1_ids) & tpu_consumed, set())

        # Hop 2: Resume GPU from Hop 1 GPU checkpoint for the remaining 2 steps (8 sequences)
        gpu_out2 = self.root / "gpu_hop2"
        p2 = subprocess.run([
            sys.executable, "-m", "tot_360m.src.train_gpu_calibration",
            "--config", "mini", "--device", "cpu", "--fixture", "--precision", "fp32",
            "--data-dir", str(data), "--output-dir", str(gpu_out2),
            "--seq-len", "16", "--batch-size", "4", "--grad-accum-steps", "1",
            "--max-tokens", "50000", "--max-steps", "2", "--schedule-steps", "10",
            "--max-seconds", "180", "--save-margin", "5", "--save-interval", "1", "--warmup-steps", "1",
            "--record-batch-ids", "--resume", str(gpu_out1 / "final.pt"),
            "--allow-bulk-manifest", "--skip-checksum-verify",
        ], capture_output=True, text=True, env=env, timeout=180)
        self.assertEqual(p2.returncode, 0, p2.stdout + p2.stderr)
        sum2 = json.loads((gpu_out2 / "summary.json").read_text())
        self.assertEqual(sum2["steps_this_run"], 2)
        self.assertEqual(sum2["global_step"], 54)
        self.assertEqual(sum2["remaining_unique_sequences"], 0)
        hop2_ids = [tuple(x) for x in sum2["batch_ids"]]
        self.assertEqual(len(hop2_ids), 8)
        # Together, TPU (16) + GPU Hop 1 (8) + GPU Hop 2 (8) must cover all 32 sequences with zero duplicates
        all_32 = tpu_consumed | set(hop1_ids) | set(hop2_ids)
        self.assertEqual(len(all_32), 32)


if __name__ == "__main__":
    unittest.main()
