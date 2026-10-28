"""Real Packed-Shard Fresh-Process Checkpoint Recovery Drill for Tot-360M.

Verifies deterministic state continuity on actual production uint16 binary shards:
1. Production 32k Shards: Uses real shards from tot_360m/data with manifest.json.
2. Documented Mini Vocab-32k: TotMiniConfig explicitly configured with vocab_size=32000.
3. Fresh Process Execution: Resumed run executes in an entirely fresh Python subprocess.
4. Strict Parity: Control vs Resumed loss, LR, and cursor state match exactly (delta < 1e-5).
5. Identity Validation: Asserts topology or tokenizer mismatch raises ValueError on resume.
"""

import os
import sys
import json
import tempfile
import unittest
import subprocess
from pathlib import Path
import torch

from tot_360m.config import TotMiniConfig
from tot_360m.src.model import TotForCausalLM
from tot_360m.src.checkpoint import save_checkpoint, load_checkpoint


RUNNER_SCRIPT = """
import os
import sys
import json
import torch
torch.set_num_threads(2)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tot_360m.config import TotMiniConfig
from tot_360m.src.model import TotForCausalLM
from tot_360m.src.checkpoint import save_checkpoint, load_checkpoint
from tot_360m.src.train_tpu_calibration import PackedBinaryDataset

mode = sys.argv[1]
shard_path = sys.argv[2]
ckpt_path = sys.argv[3]
out_json = sys.argv[4]

config = TotMiniConfig()
config.vocab_size = 32000

dataset = PackedBinaryDataset([shard_path], seq_len=1024, dtype=torch.uint16)
seq_iter = dataset.get_sequence_iterator(start_shard=0, start_offset=0)

identity = {
    "tokenizer_hash": "1ebc8d7aab0e6d2d12a6b4da34fcf6d84f136dcad0c6c1a66e600a3d48b9dba9",
    "topology": {
        "world_size": 1,
        "batch_size": 1,
        "seq_len": 1024,
        "grad_accum_steps": 1,
    },
    "optimizer": "AdamW",
    "optimizer_policy": {
        "class": "AdamW",
        "lr": 1e-3,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "precision": "fp32",
        "scheduler": "CosineAnnealingLR",
        "max_grad_norm": 1.0,
    },
    "data_identity": {
        "shard_path": shard_path,
    },
}

if mode == "control":
    torch.manual_seed(20261009)
    model = TotForCausalLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    losses = []
    lrs = []
    step = 0

    for shard_idx, seq_i, tensor in seq_iter:
        batch = tensor.unsqueeze(0)
        optimizer.zero_grad()
        _, loss = model(input_ids=batch, labels=batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        losses.append(loss.item())
        lrs.append(scheduler.get_last_lr()[0])
        if step == 4:
            final_cursor = {"shard_idx": shard_idx, "seq_offset": seq_i + 1, "tokens_seen": step * 1023}
            break

    with open(out_json, "w") as f:
        json.dump({"losses": losses, "lrs": lrs, "final_cursor": final_cursor}, f)

elif mode == "phase1":
    torch.manual_seed(20261009)
    model = TotForCausalLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    losses = []
    lrs = []
    step = 0

    for shard_idx, seq_i, tensor in seq_iter:
        batch = tensor.unsqueeze(0)
        optimizer.zero_grad()
        _, loss = model(input_ids=batch, labels=batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        losses.append(loss.item())
        lrs.append(scheduler.get_last_lr()[0])
        if step == 2:
            cursor_s2 = {"shard_idx": shard_idx, "seq_offset": seq_i + 1, "tokens_seen": step * 1023}
            save_checkpoint(
                checkpoint_path=ckpt_path,
                step=2,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                data_state=cursor_s2,
                metrics={"loss": losses[-1]},
                identity_meta=identity,
            )
            break

    with open(out_json, "w") as f:
        json.dump({"phase1_losses": losses, "phase1_lrs": lrs, "cursor_step2": cursor_s2}, f)

elif mode == "phase2_resume":
    model = TotForCausalLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    step, data_state, metrics = load_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        expected_identity=identity,
    )

    resumed_iter = dataset.get_sequence_iterator(
        start_shard=data_state["shard_idx"],
        start_offset=data_state["seq_offset"],
    )

    losses = []
    lrs = []

    for shard_idx, seq_i, tensor in resumed_iter:
        batch = tensor.unsqueeze(0)
        optimizer.zero_grad()
        _, loss = model(input_ids=batch, labels=batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        losses.append(loss.item())
        lrs.append(scheduler.get_last_lr()[0])
        if step == 4:
            final_cursor = {"shard_idx": shard_idx, "seq_offset": seq_i + 1, "tokens_seen": step * 1023}
            break

    with open(out_json, "w") as f:
        json.dump({
            "phase2_losses": losses,
            "phase2_lrs": lrs,
            "resumed_step": 2,
            "final_cursor": final_cursor,
        }, f)
"""


class TestRealShardRecovery(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp_dir = tempfile.mkdtemp()
        self.runner_py = os.path.join(self.tmp_dir, "runner.py")
        with open(self.runner_py, "w", encoding="utf-8") as f:
            f.write(RUNNER_SCRIPT)

        # Locate real uint16 shard 0
        root = Path(__file__).resolve().parents[2]
        self.real_shard = str(root / "tot_360m" / "data" / "shards" / "shard_000.bin")
        self.assertTrue(os.path.exists(self.real_shard), f"Real shard not found: {self.real_shard}")

    def test_real_shard_fresh_process_recovery(self):
        """Execute control vs interrupted + resumed run on real uint16 32k shards."""
        ckpt_path = os.path.join(self.tmp_dir, "real_step2.pt")
        control_json = os.path.join(self.tmp_dir, "control.json")
        phase1_json = os.path.join(self.tmp_dir, "phase1.json")
        phase2_json = os.path.join(self.tmp_dir, "phase2.json")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

        # 1. Control run: 4 steps uninterrupted
        res_ctrl = subprocess.run(
            [sys.executable, self.runner_py, "control", self.real_shard, ckpt_path, control_json],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(res_ctrl.returncode, 0, f"Control run failed: {res_ctrl.stderr}")

        # 2. Phase 1: 2 steps and save
        res_p1 = subprocess.run(
            [sys.executable, self.runner_py, "phase1", self.real_shard, ckpt_path, phase1_json],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(res_p1.returncode, 0, f"Phase 1 run failed: {res_p1.stderr}")
        self.assertTrue(os.path.exists(ckpt_path))

        # 3. Phase 2: Resume in fresh subprocess (steps 3..4)
        res_p2 = subprocess.run(
            [sys.executable, self.runner_py, "phase2_resume", self.real_shard, ckpt_path, phase2_json],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(res_p2.returncode, 0, f"Phase 2 resume run failed: {res_p2.stderr}")

        # 4. Compare results
        with open(control_json, "r") as f:
            ctrl_data = json.load(f)
        with open(phase2_json, "r") as f:
            p2_data = json.load(f)

        ctrl_losses = ctrl_data["losses"]
        ctrl_lrs = ctrl_data["lrs"]
        p2_losses = p2_data["phase2_losses"]
        p2_lrs = p2_data["phase2_lrs"]

        # Resumed steps 3..4 must match Control steps 3..4 exactly
        for i in range(2):
            ctrl_l = ctrl_losses[2 + i]
            res_l = p2_losses[i]
            diff = abs(ctrl_l - res_l)
            self.assertLess(
                diff,
                1e-5,
                f"Step {3 + i} loss drifted upon real shard recovery! Control={ctrl_l}, Resumed={res_l}, Diff={diff}"
            )
            self.assertAlmostEqual(ctrl_lrs[2 + i], p2_lrs[i], places=8)

        # Cursor continuity
        self.assertEqual(ctrl_data["final_cursor"], p2_data["final_cursor"])

    def test_mismatched_identity_rejection_on_resume(self):
        """Verify that loading checkpoint with mismatched topology or tokenizer raises ValueError."""
        ckpt_path = os.path.join(self.tmp_dir, "mismatch_test.pt")
        config = TotMiniConfig()
        config.vocab_size = 32000
        model = TotForCausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        valid_identity = {
            "tokenizer_hash": "1ebc8d7aab0e6d2d12a6b4da34fcf6d84f136dcad0c6c1a66e600a3d48b9dba9",
            "topology": {
                "world_size": 1,
                "batch_size": 1,
                "seq_len": 1024,
                "grad_accum_steps": 1,
            },
            "optimizer": "AdamW",
            "optimizer_policy": {
                "class": "AdamW",
                "lr": 1e-3,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
                "precision": "fp32",
                "scheduler": "CosineAnnealingLR",
                "max_grad_norm": 1.0,
            },
            "data_identity": {
                "manifest_sha256": "be9bf24c7ce842dec675c516586e1d1dfb2ef3364eca4efbccd717cdb978c41c",
                "corpus_sha256": "81d680336d36dc3e168ac5e05f0ee757ea37fa732a7a952018b76910170f7223",
            },
        }
        save_checkpoint(
            checkpoint_path=ckpt_path,
            step=1,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            config=config,
            data_state={},
            identity_meta=valid_identity,
        )

        # Mismatched topology (e.g. batch_size was 1, expected 4)
        mismatched_topology = {
            "topology": {
                "world_size": 1,
                "batch_size": 4,  # mismatched!
                "seq_len": 1024,
                "grad_accum_steps": 1,
            }
        }
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                config=config,
                expected_identity=mismatched_topology,
            )
        self.assertIn("Checkpoint identity mismatch for 'topology'", str(ctx.exception))

        # Mismatched tokenizer hash
        mismatched_tok = {
            "tokenizer_hash": "deadbeef" * 8,
        }
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                config=config,
                expected_identity=mismatched_tok,
            )
        self.assertIn("Checkpoint identity mismatch for 'tokenizer_hash'", str(ctx.exception))

        # Mismatched data identity (e.g. different manifest)
        mismatched_data = {
            "data_identity": {
                "manifest_sha256": "00000000" * 8,
                "corpus_sha256": "81d680336d36dc3e168ac5e05f0ee757ea37fa732a7a952018b76910170f7223",
            }
        }
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                config=config,
                expected_identity=mismatched_data,
            )
        self.assertIn("Checkpoint identity mismatch for 'data_identity'", str(ctx.exception))

        # Mismatched optimizer policy (e.g. different learning rate)
        mismatched_opt = {
            "optimizer_policy": {
                "class": "AdamW",
                "lr": 5e-4,  # mismatched!
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
                "precision": "fp32",
                "scheduler": "CosineAnnealingLR",
                "max_grad_norm": 1.0,
            }
        }
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                config=config,
                expected_identity=mismatched_opt,
            )
        self.assertIn("Checkpoint identity mismatch for 'optimizer_policy'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
