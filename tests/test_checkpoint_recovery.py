"""Fresh-Process Checkpoint Recovery Drill for Tot-360M.

Tests deterministic state continuity across simulated preemption boundaries:
1. Dynamic LR Scheduler: CosineAnnealingLR verified across steps.
2. RNG-Driven Batches: Batches generated dynamically inside the training loop using active RNG.
3. Dataset Cursor State: Tracks shard index, token offset, and tokens seen across save/restore.
4. Fresh Process Execution: Resumed run executes in an entirely fresh sub-process with bounded timeouts.
5. Strict Parity: Asserts loss, LR, and cursor state match uninterrupted control run (delta < 1e-6).
"""

import os
import sys
import json
import tempfile
import unittest
import subprocess
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tot_360m.config import TotMiniConfig
from tot_360m.src.model import TotForCausalLM
from tot_360m.src.checkpoint import save_checkpoint, load_checkpoint, get_config_fingerprint


RUNNER_SCRIPT = """
import os
import sys
import json
import torch

# Limit CPU threads to 2 for laptop safety
torch.set_num_threads(2)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tot_360m.config import TotMiniConfig
from tot_360m.src.model import TotForCausalLM
from tot_360m.src.checkpoint import save_checkpoint, load_checkpoint

mode = sys.argv[1]
ckpt_path = sys.argv[2]
out_json = sys.argv[3]

config = TotMiniConfig()

class SyntheticDatasetCursor:
    def __init__(self, shard_idx=0, token_offset=0, tokens_seen=0):
        self.shard_idx = shard_idx
        self.token_offset = token_offset
        self.tokens_seen = tokens_seen

    def get_batch(self, batch_size=2, seq_len=8, vocab_size=1000):
        batch = torch.randint(0, vocab_size, (batch_size, seq_len))
        self.token_offset += batch_size * seq_len
        self.tokens_seen += batch_size * seq_len
        return batch

    def state_dict(self):
        return {
            "shard_idx": self.shard_idx,
            "token_offset": self.token_offset,
            "tokens_seen": self.tokens_seen,
        }

    def load_state_dict(self, state):
        self.shard_idx = state["shard_idx"]
        self.token_offset = state["token_offset"]
        self.tokens_seen = state["tokens_seen"]

if mode == "control":
    torch.manual_seed(1337)
    model = TotForCausalLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    cursor = SyntheticDatasetCursor()

    losses = []
    lrs = []

    for step in range(1, 7):
        optimizer.zero_grad()
        batch = cursor.get_batch(vocab_size=config.vocab_size)
        _, loss = model(input_ids=batch, labels=batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        lrs.append(scheduler.get_last_lr()[0])

    with open(out_json, "w") as f:
        json.dump({"losses": losses, "lrs": lrs, "final_cursor": cursor.state_dict()}, f)

elif mode == "phase1":
    torch.manual_seed(1337)
    model = TotForCausalLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    cursor = SyntheticDatasetCursor()

    losses = []
    lrs = []

    for step in range(1, 4):
        optimizer.zero_grad()
        batch = cursor.get_batch(vocab_size=config.vocab_size)
        _, loss = model(input_ids=batch, labels=batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        lrs.append(scheduler.get_last_lr()[0])

    save_checkpoint(
        checkpoint_path=ckpt_path,
        step=3,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        data_state=cursor.state_dict(),
        metrics={"loss": losses[-1]},
    )

    with open(out_json, "w") as f:
        json.dump({"phase1_losses": losses, "phase1_lrs": lrs, "cursor_step3": cursor.state_dict()}, f)

elif mode == "phase2_resume":
    model = TotForCausalLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    cursor = SyntheticDatasetCursor()

    step, data_state, metrics = load_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    cursor.load_state_dict(data_state)

    losses = []
    lrs = []

    for s in range(step + 1, 7):
        optimizer.zero_grad()
        batch = cursor.get_batch(vocab_size=config.vocab_size)
        _, loss = model(input_ids=batch, labels=batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        lrs.append(scheduler.get_last_lr()[0])

    with open(out_json, "w") as f:
        json.dump({
            "phase2_losses": losses,
            "phase2_lrs": lrs,
            "resumed_step": step,
            "final_cursor": cursor.state_dict()
        }, f)
"""


class TestCheckpointRecovery(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp_dir = tempfile.mkdtemp()
        self.runner_py = os.path.join(self.tmp_dir, "runner.py")
        with open(self.runner_py, "w", encoding="utf-8") as f:
            f.write(RUNNER_SCRIPT)

    def test_fresh_process_exact_checkpoint_recovery(self):
        """Execute control vs interrupted + resumed run with dynamic scheduler, RNG, and cursor."""
        ckpt_path = os.path.join(self.tmp_dir, "test_step3.pt")
        control_json = os.path.join(self.tmp_dir, "control.json")
        phase1_json = os.path.join(self.tmp_dir, "phase1.json")
        phase2_json = os.path.join(self.tmp_dir, "phase2.json")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

        # 1. Run uninterrupted control with 30s timeout
        res_ctrl = subprocess.run(
            [sys.executable, self.runner_py, "control", ckpt_path, control_json],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(res_ctrl.returncode, 0, f"Control run failed: {res_ctrl.stderr}")

        # 2. Run phase 1 (steps 1..3 and save) with 30s timeout
        res_p1 = subprocess.run(
            [sys.executable, self.runner_py, "phase1", ckpt_path, phase1_json],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(res_p1.returncode, 0, f"Phase 1 run failed: {res_p1.stderr}")
        self.assertTrue(os.path.exists(ckpt_path), "Checkpoint file was not created!")

        # 3. Run phase 2 in a completely fresh sub-process (steps 4..6) with 30s timeout
        res_p2 = subprocess.run(
            [sys.executable, self.runner_py, "phase2_resume", ckpt_path, phase2_json],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(res_p2.returncode, 0, f"Phase 2 resume run failed: {res_p2.stderr}")

        # 4. Compare results
        with open(control_json, "r") as f:
            ctrl_data = json.load(f)
            control_losses = ctrl_data["losses"]
            control_lrs = ctrl_data["lrs"]
            control_cursor = ctrl_data["final_cursor"]

        with open(phase1_json, "r") as f:
            p1_losses = json.load(f)["phase1_losses"]

        with open(phase2_json, "r") as f:
            p2_data = json.load(f)
            p2_losses = p2_data["phase2_losses"]
            p2_lrs = p2_data["phase2_lrs"]
            p2_cursor = p2_data["final_cursor"]

        # Steps 1..3 must match
        for i in range(3):
            self.assertAlmostEqual(
                control_losses[i],
                p1_losses[i],
                places=6,
                msg=f"Step {i+1} loss mismatch between control and phase 1"
            )

        # Steps 4..6 (resumed in fresh process) must match uninterrupted control exactly
        for i in range(3):
            ctrl_loss = control_losses[3 + i]
            resumed_loss = p2_losses[i]
            diff = abs(ctrl_loss - resumed_loss)
            self.assertLess(
                diff,
                1e-6,
                f"Step {4 + i} loss drifted upon checkpoint recovery! Control={ctrl_loss}, Resumed={resumed_loss}, Diff={diff}"
            )

            # Assert learning rate continuity
            ctrl_lr = control_lrs[3 + i]
            resumed_lr = p2_lrs[i]
            self.assertAlmostEqual(ctrl_lr, resumed_lr, places=8, msg=f"Step {4+i} LR mismatch!")

        # Assert data cursor state continuity
        self.assertEqual(p2_data["resumed_step"], 3)
        self.assertEqual(control_cursor["tokens_seen"], p2_cursor["tokens_seen"])
        self.assertEqual(control_cursor["token_offset"], p2_cursor["token_offset"])

    def test_incompatible_config_rejection(self):
        """Verify that loading a checkpoint with mismatched config fingerprint raises ValueError."""
        ckpt_path = os.path.join(self.tmp_dir, "mismatch_test.pt")
        config_mini = TotMiniConfig()
        model = TotForCausalLM(config_mini)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        save_checkpoint(
            checkpoint_path=ckpt_path,
            step=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config_mini,
            data_state={},
        )

        from tot_360m.config import Tot360MConfig
        config_full = Tot360MConfig()
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config_full,
            )
        self.assertIn("fingerprint mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
