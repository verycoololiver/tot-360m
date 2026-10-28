"""Deterministic Checkpoint Manager for Tot-360M Pretraining.

Guarantees exact, loss-free resumption across preemption boundaries on cloud TPU/GPU:
- Full structural fingerprint: vocab_size, hidden_size, intermediate_size, num_hidden_layers,
  num_attention_heads, num_key_value_heads, head_dim, rope_theta, rms_norm_eps,
  max_position_embeddings, z_loss_weight, use_qk_norm, tie_word_embeddings, torch_version.
- Captures model weights, AdamW optimizer moments, LR scheduler state.
- Captures full RNG states across Python random, NumPy, and PyTorch (CPU/CUDA).
- Captures dataset cursor state: shard_idx, token_offset, total_tokens_seen, epoch.
- Uses atomic file swaps to eliminate partial/corrupted write hazards.
"""

import os
import gc
import random
import hashlib
import json
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from tot_360m.config import Tot360MConfig

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def get_config_fingerprint(config: Tot360MConfig, optimizer_cls_name: Optional[str] = None) -> str:
    """Computes a deterministic hash of all model and optimizer hyperparameters."""
    config_dict = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "rope_theta": float(config.rope_theta),
        "rms_norm_eps": float(config.rms_norm_eps),
        "max_position_embeddings": config.max_position_embeddings,
        "z_loss_weight": float(config.z_loss_weight),
        "use_qk_norm": config.use_qk_norm,
        "tie_word_embeddings": config.tie_word_embeddings,
        "torch_version": torch.__version__,
        "optimizer": optimizer_cls_name or "AdamW",
    }
    encoded = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def capture_rng_states() -> Dict[str, Any]:
    """Captures all active pseudo-random number generator states."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def _rng_tuple(value: Any) -> Any:
    """Restore sequence types changed by XLA's CPU serialization rewrite."""
    if isinstance(value, (tuple, list)):
        return tuple(_rng_tuple(item) for item in value)
    return value


def restore_rng_states(states: Dict[str, Any]):
    """Restores pseudo-random number generator states precisely."""
    if "python" in states:
        # xm.save turns nested tuples into lists. Python requires both the
        # outer state and its internal MT state vector to remain tuples.
        random.setstate(_rng_tuple(states["python"]))
    if "numpy" in states:
        np.random.set_state(_rng_tuple(states["numpy"]))
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def save_checkpoint(
    checkpoint_path: str,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    config: Tot360MConfig,
    data_state: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    identity_meta: Optional[Dict[str, Any]] = None,
):
    """Atomically saves a full training checkpoint."""
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    os.makedirs(checkpoint_dir, exist_ok=True)
    temp_path = f"{checkpoint_path}.tmp"

    optimizer_name = optimizer.__class__.__name__ if optimizer is not None else None

    payload = {
        "version": 2,
        "step": step,
        "config_fingerprint": get_config_fingerprint(config, optimizer_name),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_states": capture_rng_states(),
        "data_state": data_state,
        "metrics": metrics or {},
        "identity_meta": identity_meta or {},
    }

    # Save to temp file and atomically swap to target path
    if HAS_XLA:
        xm.mark_step(wait=True)
        xm.save(payload, temp_path, master_only=True, global_master=True)
        if xm.is_master_ordinal(local=False):
            os.replace(temp_path, checkpoint_path)
        xm.rendezvous(f"checkpoint-written-{step}")
    else:
        torch.save(payload, temp_path)
        os.replace(temp_path, checkpoint_path)
    del payload
    gc.collect()


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    config: Optional[Tot360MConfig] = None,
    expected_identity: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, Any], Dict[str, float]]:
    """Validates and loads a training checkpoint, restoring model, optimizer, scheduler, and RNG states."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 1. Structural compatibility check
    is_gpu_ckpt = checkpoint.get("format") == "tot-gpu-candidate-v1"
    if config is not None:
        if is_gpu_ckpt and "config_fingerprint" not in checkpoint:
            gpu_cfg = checkpoint.get("identity", {}).get("config", {})
            for field in (
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
            ):
                if field in gpu_cfg and gpu_cfg[field] != getattr(config, field):
                    raise ValueError(
                        f"GPU checkpoint config mismatch on '{field}': expected {getattr(config, field)}, found {gpu_cfg[field]}"
                    )
        else:
            opt_name = optimizer.__class__.__name__ if optimizer is not None else None
            expected_fp = get_config_fingerprint(config, opt_name)
            actual_fp = checkpoint.get("config_fingerprint")
            if actual_fp != expected_fp:
                raise ValueError(
                    f"Checkpoint configuration fingerprint mismatch! Expected {expected_fp}, found {actual_fp}"
                )

    # 1b. Identity metadata validation (tokenizer, manifest, topology, optimizer)
    if expected_identity is not None:
        actual_ident = checkpoint.get("identity_meta", {})
        for key, exp_val in expected_identity.items():
            act_val = actual_ident.get(key)
            if act_val is None:
                raise ValueError(f"Checkpoint identity metadata missing required field: '{key}'")
            if act_val != exp_val:
                raise ValueError(
                    f"Checkpoint identity mismatch for '{key}'! Expected {exp_val}, found {act_val}"
                )

    # 2. Restore model state and preserve weight tying across device moves/loads
    model_state = checkpoint.pop("model_state")
    model.load_state_dict(model_state)
    del model_state
    if getattr(config, "tie_word_embeddings", True) and hasattr(model, "lm_head") and hasattr(model, "model"):
        model.lm_head.weight = model.model.embed_tokens.weight
    if HAS_XLA:
        xm.mark_step()

    # 3. Restore optimizer state
    opt_state = checkpoint.pop("optimizer_state", None)
    if optimizer is not None and opt_state is not None:
        optimizer.load_state_dict(opt_state)
        if HAS_XLA:
            xm.mark_step()
    del opt_state

    # 4. Restore scheduler state
    sched_state = checkpoint.pop("scheduler_state", None)
    if scheduler is not None and sched_state is not None:
        scheduler.load_state_dict(sched_state)
    del sched_state

    # 5. Restore RNG states
    if "rng_states" in checkpoint:
        restore_rng_states(checkpoint.pop("rng_states"))
    elif is_gpu_ckpt and checkpoint.get("rank_states"):
        restore_rng_states(checkpoint["rank_states"][0])

    step = checkpoint.get("step", 0)
    if is_gpu_ckpt and "data_state" not in checkpoint:
        data_state = {
            "tokens_seen": checkpoint.get("tokens_seen", 0),
            "global_seqs_consumed_in_pack": int(checkpoint.get("cursor", 0)),
        }
    else:
        data_state = checkpoint.get("data_state", {})
    metrics = checkpoint.get("metrics", {})
    del checkpoint
    gc.collect()

    return step, data_state, metrics
