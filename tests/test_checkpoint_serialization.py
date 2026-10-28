"""Regression tests for the real v6 XLA tuple-to-list checkpoint failure.

The fresh-process test emulates XLA's container rewrite on CPU. It is not
evidence of TPU resume parity; that gate remains in the cloud notebook.
"""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from tot_360m.src.checkpoint import capture_rng_states, restore_rng_states


def xla_containers(value):
    if isinstance(value, dict):
        return {key: xla_containers(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [xla_containers(item) for item in value]
    return value


RUNNER = r'''
import json, random, sys
import numpy as np
import torch
torch.set_num_threads(2)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tot_360m.config import TotMiniConfig
from tot_360m.src.model import TotForCausalLM
from tot_360m.src.checkpoint import save_checkpoint, load_checkpoint
from tot_360m.tests.test_checkpoint_serialization import xla_containers

mode, checkpoint_path, output_path, serialization = sys.argv[1:]
random.seed(1337); np.random.seed(1337); torch.manual_seed(1337)
config = TotMiniConfig()
model = TotForCausalLM(config)
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=6)
identity = {'topology': {'world_size': 1}, 'tokenizer': 'fixture',
            'optimizer_policy': {'betas': [0.9, 0.999]}}
cursor = {'tokens_seen': 0, 'seq_offset': 0}
start = 0
if mode == 'resume':
    # Construction consumes RNG in this fresh process; load must reset it.
    start, cursor, _ = load_checkpoint(checkpoint_path, model, optimizer,
                                       scheduler, config, identity)
elif mode != 'control' and mode != 'phase1':
    raise ValueError(mode)
losses = []
limit = 3 if mode == 'phase1' else 6
for step in range(start + 1, limit + 1):
    shift = random.randrange(1000) + int(np.random.randint(0, 1000))
    batch = (torch.randint(0, 1000, (1, 8)) + shift) % 1000
    optimizer.zero_grad()
    _, loss = model(batch, labels=batch)
    loss.backward(); optimizer.step(); scheduler.step()
    cursor = {'tokens_seen': cursor['tokens_seen'] + 7,
              'seq_offset': cursor['seq_offset'] + 1}
    losses.append(float(loss.detach()))
save_checkpoint(checkpoint_path, limit, model, optimizer, scheduler, config,
                cursor, {'loss': losses[-1]}, identity)
if serialization == 'xla':
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    torch.save(xla_containers(payload), checkpoint_path)
with open(output_path, 'w', encoding='utf-8') as stream:
    json.dump({'losses': losses, 'cursor': cursor, 'start_step': start,
               'lr': scheduler.get_last_lr()}, stream)
'''


class TestCheckpointSerialization(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.previous_rng = capture_rng_states()

    def tearDown(self):
        restore_rng_states(self.previous_rng)

    def _rng_roundtrip(self, serialize):
        random.seed(987); np.random.seed(654); torch.manual_seed(321)
        random.gauss(0, 1); np.random.standard_normal(1)  # exercise caches
        saved = serialize(capture_rng_states())
        expected = (random.random(), random.gauss(0, 1),
                    np.random.standard_normal(12), torch.rand(12))
        for _ in range(2):
            restore_rng_states(saved)
            self.assertEqual(expected[0], random.random())
            self.assertEqual(expected[1], random.gauss(0, 1))
            np.testing.assert_array_equal(expected[2], np.random.standard_normal(12))
            torch.testing.assert_close(expected[3], torch.rand(12), rtol=0, atol=0)
        return saved

    def test_cpu_tuple_rng_roundtrip(self):
        self._rng_roundtrip(copy.deepcopy)

    def test_xla_list_rng_roundtrip_does_not_mutate_payload(self):
        saved = self._rng_roundtrip(xla_containers)
        self.assertIsInstance(saved['python'], list)
        self.assertIsInstance(saved['python'][1], list)
        self.assertIsInstance(saved['numpy'], list)

    def test_invalid_python_rng_is_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            restore_rng_states({'python': [3, [1, 2], None]})

    def test_fresh_process_cpu_and_xla_checkpoint_recovery(self):
        with tempfile.TemporaryDirectory(prefix='tot-rng-regression-') as folder:
            root = Path(folder)
            script = root / 'runner.py'
            script.write_text(RUNNER, encoding='utf-8')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
            def run(mode, ckpt, serialization):
                output = root / (ckpt.stem + '-' + mode + '.json')
                result = subprocess.run([sys.executable, str(script), mode,
                    str(ckpt), str(output), serialization], env=env,
                    capture_output=True, text=True, timeout=300)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(output.read_text(encoding='utf-8'))
            control_ckpt = root / 'control.pt'
            control = run('control', control_ckpt, 'cpu')
            control_state = torch.load(control_ckpt, weights_only=False)
            for serialization in ('cpu', 'xla'):
                with self.subTest(serialization=serialization):
                    candidate = root / (serialization + '.pt')
                    phase1 = run('phase1', candidate, serialization)
                    resumed = run('resume', candidate, serialization)
                    self.assertEqual(resumed['start_step'], 3)
                    self.assertEqual(control['losses'], phase1['losses'] + resumed['losses'])
                    self.assertEqual(control['cursor'], resumed['cursor'])
                    self.assertEqual(control['lr'], resumed['lr'])
                    resumed_state = torch.load(candidate, weights_only=False)
                    for name, tensor in control_state['model_state'].items():
                        torch.testing.assert_close(tensor, resumed_state['model_state'][name], rtol=0, atol=0)
                    for parameter, slot in control_state['optimizer_state']['state'].items():
                        for name, value in slot.items():
                            torch.testing.assert_close(value, resumed_state['optimizer_state']['state'][parameter][name], rtol=0, atol=0)
                    self.assertEqual(control_state['scheduler_state'], resumed_state['scheduler_state'])
                    self.assertEqual(control_state['data_state'], resumed_state['data_state'])


if __name__ == '__main__':
    unittest.main()
