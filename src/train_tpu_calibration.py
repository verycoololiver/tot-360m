"""Kaggle TPU v3-8 Calibration & Pretraining Runner for Tot-360M.

Features:
- Bounded Execution: strictly capped at max 100M valid prediction targets or 2 hours session time
- Multi-Core TPU v3-8 (8 cores) support via PyTorch XLA (xmp.spawn) with CPU/CUDA fallback
- Static-Shape Execution: strictly fixed [batch_size, seq_len] tensors with zero padding waste
- Rank-Disjoint Shard Distribution: each core processes unique, non-overlapping packed binary shards
- Coordinated Atomic Checkpointing: saves model, optimizer, scheduler, RNG, per-rank cursor, and config fingerprints
- Accurate Metrics: accounts for entire accumulated step, excludes warmup from steady-state rate
- Finite checks and gradient clipping: stops on non-finite loss/gradients
- Clean Auto-Exit: persists metrics and checkpoints, then shuts down cleanly (sys.exit(0))
"""

import os
import sys
import time
import json
import glob
import math
import argparse
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Check for PyTorch XLA
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    try:
        import torch_xla.runtime as xr
    except ImportError:
        xr = None
    HAS_XLA = True
except ImportError:
    HAS_XLA = False
    xr = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tot_360m.config import Tot360MConfig, TotMiniConfig
from tot_360m.src.model import TotForCausalLM
from tot_360m.src.checkpoint import get_config_fingerprint, save_checkpoint, load_checkpoint


class PackedBinaryDataset:
    """Memory-efficient reader for fixed-length packed token sequences stored in flat binary format."""
    def __init__(self, shard_paths: List[str], seq_len: int = 1024, dtype=torch.uint16):
        self.shard_paths = sorted(shard_paths)
        self.seq_len = seq_len
        self.dtype = dtype
        self.bytes_per_token = 2 if dtype == torch.uint16 else 4
        self.bytes_per_seq = seq_len * self.bytes_per_token

        # Index shards and sequence counts
        self.shard_seq_counts = []
        for p in self.shard_paths:
            size = os.path.getsize(p)
            self.shard_seq_counts.append(size // self.bytes_per_seq)

        self.total_seqs = sum(self.shard_seq_counts)

    def get_sequence_iterator(self, start_shard: int = 0, start_offset: int = 0):
        """Yields (shard_idx, seq_i, tensor) starting from given cursor."""
        for shard_idx in range(start_shard, len(self.shard_paths)):
            path = self.shard_paths[shard_idx]
            offset = start_offset if shard_idx == start_shard else 0
            file_size = os.path.getsize(path)
            num_seqs = file_size // self.bytes_per_seq

            if offset >= num_seqs:
                continue

            with open(path, "rb") as f:
                f.seek(offset * self.bytes_per_seq)
                for seq_i in range(offset, num_seqs):
                    buf = f.read(self.bytes_per_seq)
                    if len(buf) < self.bytes_per_seq:
                        break
                    tensor = torch.frombuffer(bytearray(buf), dtype=self.dtype).to(torch.int64)
                    yield shard_idx, seq_i, tensor


def generate_synthetic_shards_for_testing(
    data_dir: str,
    num_shards: int = 8,
    seqs_per_shard: int = 12500,
    seq_len: int = 1024,
    vocab_size: int = 32000,
    dtype=torch.int32,
) -> List[str]:
    """Generates synthetic packed binary shards strictly for local unit/smoke testing."""
    os.makedirs(data_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(data_dir, "test_shard_*.bin")))
    if existing:
        return existing

    print(f"[TEST FIXTURE] Generating {num_shards} synthetic packed binary shards...")
    torch.manual_seed(20261009)
    shards = []
    for s_idx in range(num_shards):
        shard_path = os.path.join(data_dir, f"test_shard_{s_idx:03d}.bin")
        tokens = torch.randint(0, vocab_size, (seqs_per_shard, seq_len), dtype=dtype)
        with open(shard_path, "wb") as f:
            f.write(tokens.numpy().tobytes())
        shards.append(shard_path)
    return shards


def compute_all_rank_cursors(
    all_shards: List[str],
    world_size: int,
    seq_len: int,
    seqs_consumed_per_rank: int,
    dtype=torch.int32,
) -> Dict[str, Dict[str, int]]:
    """Computes deterministic next cursor (shard_idx, seq_offset) for EVERY rank."""
    rank_cursors = {}
    bytes_per_token = 2 if dtype == torch.uint16 else 4
    bytes_per_seq = seq_len * bytes_per_token

    for r in range(world_size):
        r_shards = [s for i, s in enumerate(all_shards) if (i % world_size) == r]
        rem = seqs_consumed_per_rank
        s_idx = 0
        for idx, s_path in enumerate(r_shards):
            file_size = os.path.getsize(s_path)
            count = file_size // bytes_per_seq
            if rem < count:
                s_idx = idx
                break
            else:
                rem -= count
                s_idx = idx + 1
        rank_cursors[str(r)] = {"shard_idx": s_idx, "seq_offset": rem}
    return rank_cursors


def compute_rank_remaining_sequences(
    shards: List[str],
    shard_idx: int,
    seq_offset: int,
    seq_len: int,
    dtype=torch.int32,
) -> int:
    """Computes exact remaining sequences for a rank starting from (shard_idx, seq_offset)."""
    bytes_per_token = 2 if dtype == torch.uint16 else 4
    bytes_per_seq = seq_len * bytes_per_token
    remaining = 0
    for idx, s_path in enumerate(shards):
        file_size = os.path.getsize(s_path)
        count = file_size // bytes_per_seq
        if idx < shard_idx:
            continue
        elif idx == shard_idx:
            remaining += max(0, count - seq_offset)
        else:
            remaining += count
    return remaining


def compute_rank_consumed_sequences(
    shards: List[str],
    shard_idx: int,
    seq_offset: int,
    seq_len: int,
    dtype=torch.int32,
) -> int:
    """Computes exact sequences already consumed by a rank up to (shard_idx, seq_offset)."""
    bytes_per_token = 2 if dtype == torch.uint16 else 4
    bytes_per_seq = seq_len * bytes_per_token
    consumed = 0
    for idx, s_path in enumerate(shards):
        file_size = os.path.getsize(s_path)
        count = file_size // bytes_per_seq
        if idx < shard_idx:
            consumed += count
        elif idx == shard_idx:
            consumed += min(count, max(0, seq_offset))
            break
        else:
            break
    return consumed


def check_coordinated_stop(local_stop: bool, device) -> bool:
    """Coordinates stop flag across all ranks to prevent collective all-reduce hangs."""
    if not HAS_XLA:
        return local_stop
    flag = torch.tensor(int(local_stop), dtype=torch.int32, device=device)
    flag = xm.all_reduce(xm.REDUCE_MAX, flag)
    xm.mark_step()
    return bool(flag.item())


def compute_file_sha256(path: str) -> str:
    """Computes SHA256 in 64KB chunks to avoid reading large files into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def train_worker(rank: int, args: argparse.Namespace, t_session_start: float):
    """Main worker executing on single TPU core (or single CPU/GPU)."""
    # 1. Device and Multi-processing Initialization
    if HAS_XLA:
        device = xm.xla_device()
        device_type = xr.device_type() if (xr is not None and hasattr(xr, "device_type")) else (xm.xla_device_hw(device) if hasattr(xm, "xla_device_hw") else "TPU")
        if xr is not None and hasattr(xr, "world_size"):
            world_size = xr.world_size()
            global_rank = xr.global_ordinal()
        else:
            world_size = xm.xrt_world_size()
            global_rank = xm.get_ordinal()
        is_master = (global_rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_type = "CUDA" if torch.cuda.is_available() else "CPU"
        world_size = 1
        global_rank = 0
        is_master = True

    # Hardware admission gate: reject CPU/GPU fallback for baseline
    if args.config == "baseline" and not getattr(args, "allow_cpu_baseline", False):
        if not HAS_XLA or device_type != "TPU":
            raise RuntimeError(
                f"Baseline 345M pretraining strictly requires genuine TPU execution (HAS_XLA={HAS_XLA}, device_type='{device_type}'). "
                f"Automatic CPU/GPU fallback is forbidden. Supply --allow-cpu-baseline only for explicit testing."
            )
        if world_size < 8:
            raise RuntimeError(
                f"Baseline 345M pretraining requires 8 TPU replicas, found world_size={world_size}."
            )

    # Synchronized deterministic initialization across all replicas
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if is_master:
        print(f"=== Tot-360M Calibration Run ===")
        print(f"Hardware Admission: Verified {device_type} on {device}")
        print(f"World Size: {world_size} replicas | Master Ordinal: {global_rank}")
        print(f"Max Targets: {args.max_tokens:,} | Max Duration: {args.max_seconds}s")
        print(f"Batch Size per core: {args.batch_size} | Seq Len: {args.seq_len}")
        print(f"Gradient Accumulation Steps: {args.grad_accum_steps}")
        print(f"Precision Mode: {getattr(args, 'precision', 'fp32')}")

    # 2. Config & Preflight Checks
    if args.config == "mini":
        config = TotMiniConfig()
        if getattr(args, "vocab_size_override", None):
            config.vocab_size = args.vocab_size_override
            if is_master:
                print(f"[CONFIG OVERRIDE] Mini model configured with explicit vocab_size={config.vocab_size}")
        elif getattr(args, "allow_tokenizer_mismatch", False) or not args.allow_synthetic_shards:
            config.vocab_size = 32000
            if is_master:
                print(f"[PREFLIGHT OVERRIDE] Mini model configured with vocab_size={config.vocab_size} to match production 32k shard tokens.")
    else:
        config = Tot360MConfig()

    if args.allow_synthetic_shards and args.config != "mini":
        raise ValueError(
            "Synthetic shards are strictly forbidden for baseline pretraining. "
            "They are permitted only for --config mini test fixtures."
        )

    # Preflight data manifest & shard validation before model allocation
    dataset_dtype = torch.int32 if args.config == "mini" and args.allow_synthetic_shards else torch.uint16
    if not args.allow_synthetic_shards:
        manifest_path = os.path.join(args.data_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Required data manifest not found at: {manifest_path}! "
                f"Real training requires verified, immutable shards with provenance."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            data_manifest = json.load(f)

        required_fields = ["tokenizer_json_sha256", "dtype", "seq_len", "vocab_size", "shards"]
        for field in required_fields:
            if field not in data_manifest or not data_manifest[field]:
                raise ValueError(f"Data manifest missing required field: {field}")

        expected_tok_hash = "1ebc8d7aab0e6d2d12a6b4da34fcf6d84f136dcad0c6c1a66e600a3d48b9dba9"
        manifest_tok_hash = data_manifest["tokenizer_json_sha256"]
        if manifest_tok_hash != expected_tok_hash and not getattr(args, "allow_tokenizer_mismatch", False):
            raise ValueError(
                f"Data manifest tokenizer hash mismatch! Expected {expected_tok_hash}, found {manifest_tok_hash}"
            )

        if data_manifest["vocab_size"] != config.vocab_size and not getattr(args, "allow_tokenizer_mismatch", False):
            raise ValueError(
                f"Data manifest vocab_size ({data_manifest['vocab_size']}) != model config ({config.vocab_size})"
            )

        if data_manifest.get("seq_len") != args.seq_len:
            raise ValueError(
                f"Data manifest seq_len ({data_manifest.get('seq_len')}) != requested seq_len ({args.seq_len})"
            )

        manifest_dtype = data_manifest["dtype"]
        if manifest_dtype not in ("uint16", "int32"):
            raise ValueError(f"Unsupported manifest dtype: {manifest_dtype}. Must be 'uint16' or 'int32'.")
        dataset_dtype = torch.uint16 if manifest_dtype == "uint16" else torch.int32
        bytes_per_seq = args.seq_len * (2 if dataset_dtype == torch.uint16 else 4)

        validated_manifest_shards = []
        for s_info in data_manifest["shards"]:
            s_name = s_info.get("name")
            if not s_name:
                raise ValueError("Manifest shard entry missing 'name' field")
            s_path = os.path.join(args.data_dir, s_name)
            if not os.path.exists(s_path):
                raise FileNotFoundError(f"Manifest declared shard not found on disk: {s_path}")

            file_size = os.path.getsize(s_path)
            if file_size == 0 or file_size % bytes_per_seq != 0:
                raise ValueError(
                    f"Shard {s_name} size ({file_size} bytes) is not aligned to sequence size ({bytes_per_seq} bytes)!"
                )

            if "sha256" not in s_info or not s_info["sha256"]:
                raise ValueError(f"Manifest shard {s_name} missing required sha256 checksum!")
            if not getattr(args, "skip_checksum_verify", False):
                actual_hash = compute_file_sha256(s_path)
                if actual_hash != s_info["sha256"]:
                    raise ValueError(f"Shard checksum mismatch for {s_name}! Expected {s_info['sha256']}, got {actual_hash}")

            # Range check: verify token IDs on sample sequence
            with open(s_path, "rb") as f:
                first_seq = f.read(bytes_per_seq)
                if len(first_seq) == bytes_per_seq:
                    sample_tokens = torch.frombuffer(bytearray(first_seq), dtype=dataset_dtype).to(torch.int64)
                    min_id = sample_tokens.min().item()
                    max_id = sample_tokens.max().item()
                    if min_id < 0 or max_id >= config.vocab_size:
                        raise ValueError(
                            f"Shard {s_name} token ID out of range [0, {config.vocab_size - 1}]: min={min_id}, max={max_id}"
                        )

            validated_manifest_shards.append(s_path)

    model = TotForCausalLM(config).to(device)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    # Broadcast master weights to guarantee identical replicas across all cores on fresh initialization.
    # When resuming from a checkpoint, all ranks load the identical checkpoint file deterministically,
    # avoiding compilation of a redundant 290-tensor collective broadcast HLO graph in TPU HBM.
    if HAS_XLA and not args.resume_ckpt:
        xm.broadcast_master_param(model)
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        xm.mark_step()

    if config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight, "Weight tying broken after device initialization!"

    # 3. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    targets_per_optimizer_step = world_size * args.batch_size * (args.seq_len - 1) * args.grad_accum_steps
    total_steps = math.ceil(args.max_tokens / max(targets_per_optimizer_step, 1))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 100), eta_min=args.lr * 0.1)

    optimizer_policy = {
        "class": optimizer.__class__.__name__,
        "lr": args.lr,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "precision": getattr(args, "precision", "fp32"),
        "scheduler": scheduler.__class__.__name__,
        "max_grad_norm": args.max_grad_norm,
    }

    expected_identity = {
        "topology": {
            "world_size": world_size,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "grad_accum_steps": args.grad_accum_steps,
        },
        "optimizer": optimizer.__class__.__name__,
        "optimizer_policy": optimizer_policy,
    }
    if not args.allow_synthetic_shards:
        manifest_sha256 = compute_file_sha256(manifest_path)
        expected_identity["data_identity"] = {
            "manifest_sha256": manifest_sha256,
            "corpus_sha256": data_manifest.get("corpus_jsonl", {}).get("sha256"),
            "num_shards": len(validated_manifest_shards),
            "shard_hashes": [s["sha256"] for s in data_manifest.get("shards", [])],
        }
        if not getattr(args, "allow_tokenizer_mismatch", False):
            expected_identity["tokenizer_hash"] = expected_tok_hash

    # 4. Data Shard Distribution (Rank-Disjoint)
    if not args.allow_synthetic_shards:
        all_shards = validated_manifest_shards
    else:
        all_shards = sorted(
            glob.glob(os.path.join(args.data_dir, "*.bin")) +
            glob.glob(os.path.join(args.data_dir, "shards", "*.bin"))
        )
        if not all_shards:
            all_shards = generate_synthetic_shards_for_testing(
                args.data_dir,
                num_shards=max(world_size * 2, 8),
                seqs_per_shard=100 if args.config == "mini" else 12500,
                seq_len=args.seq_len,
                vocab_size=config.vocab_size,
                dtype=dataset_dtype,
            )

    # Assign strictly disjoint shards to each rank
    my_shards = [s for i, s in enumerate(all_shards) if (i % world_size) == global_rank]
    if not my_shards:
        raise ValueError(f"Rank {global_rank} has 0 assigned shards! Need at least {world_size} shards.")

    dataset = PackedBinaryDataset(my_shards, seq_len=args.seq_len, dtype=dataset_dtype)

    # 5. Resume from Checkpoint if Requested
    start_step = 0
    tokens_seen = 0
    start_shard = 0
    start_offset = 0
    rank_cursors = {}

    if args.resume_ckpt:
        if not os.path.exists(args.resume_ckpt):
            raise FileNotFoundError(
                f"Requested resume checkpoint not found: {args.resume_ckpt}! "
                f"Missing checkpoint must fail explicitly rather than silently restarting."
            )
        if is_master:
            print(f"Resuming training from checkpoint: {args.resume_ckpt}")
        step, data_state, metrics = load_checkpoint(
            checkpoint_path=args.resume_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            expected_identity=None if getattr(args, "skip_identity_verify", False) else expected_identity,
        )
        start_step = step
        tokens_seen = data_state.get("tokens_seen", 0)

        # Ensure CosineAnnealingLR horizon expands if max_tokens / step budget was increased for a bulk phase,
        # and recompute closed-form cosine LR at start_step so PyTorch's recursive step() does not stay pinned near eta_min.
        remaining_tokens_at_start = max(0, args.max_tokens - tokens_seen)
        desired_total_steps = start_step + math.ceil(remaining_tokens_at_start / max(targets_per_optimizer_step, 1))
        if getattr(scheduler, "T_max", 0) < desired_total_steps:
            scheduler.T_max = max(desired_total_steps, 100)
            new_lrs = []
            for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups):
                if scheduler.last_epoch <= 0:
                    lr_val = float(base_lr)
                elif scheduler.last_epoch >= scheduler.T_max:
                    lr_val = float(scheduler.eta_min)
                else:
                    lr_val = float(
                        scheduler.eta_min
                        + (base_lr - scheduler.eta_min)
                        * (1.0 + math.cos(math.pi * scheduler.last_epoch / scheduler.T_max))
                        / 2.0
                    )
                group["lr"] = lr_val
                new_lrs.append(lr_val)
            scheduler._last_lr = new_lrs

        if getattr(args, "reset_data_cursor", False):
            rank_cursors = {}
            start_shard = 0
            start_offset = 0
        else:
            rank_cursors = data_state.get("rank_cursors", {})
            if not rank_cursors and "global_seqs_consumed_in_pack" in data_state:
                seqs_per_rank_from_global = int(data_state["global_seqs_consumed_in_pack"]) // world_size
                rank_cursors = compute_all_rank_cursors(
                    all_shards=all_shards,
                    world_size=world_size,
                    seq_len=args.seq_len,
                    seqs_consumed_per_rank=seqs_per_rank_from_global,
                    dtype=dataset.dtype,
                )
            my_cursor = rank_cursors.get(str(global_rank), {})
            start_shard = my_cursor.get("shard_idx", 0)
            start_offset = my_cursor.get("seq_offset", 0)
        if is_master:
            print(f"Resumed at step {start_step}, tokens_seen {tokens_seen:,}, rank {global_rank} cursor ({start_shard}, {start_offset}), T_max={scheduler.T_max}")

    initial_seqs_consumed_per_rank = compute_rank_consumed_sequences(
        shards=my_shards,
        shard_idx=start_shard,
        seq_offset=start_offset,
        seq_len=args.seq_len,
        dtype=dataset.dtype,
    )

    # Calculate available sequences per rank to prevent unequal shard exhaustion deadlock
    seqs_per_opt_step = args.batch_size * args.grad_accum_steps
    rank_available_seqs = []
    for r in range(world_size):
        r_shards = [s for i, s in enumerate(all_shards) if (i % world_size) == r]
        if r == global_rank:
            r_shard_idx = start_shard
            r_seq_offset = start_offset
        else:
            cur = rank_cursors.get(str(r), {})
            r_shard_idx = cur.get("shard_idx", 0)
            r_seq_offset = cur.get("seq_offset", 0)
        rem = compute_rank_remaining_sequences(
            shards=r_shards,
            shard_idx=r_shard_idx,
            seq_offset=r_seq_offset,
            seq_len=args.seq_len,
            dtype=dataset.dtype,
        )
        rank_available_seqs.append(rem)

    min_safe_seqs = min(rank_available_seqs) if rank_available_seqs else 0
    max_safe_session_steps = min_safe_seqs // seqs_per_opt_step

    # Strict token cap step bound: stop BEFORE taking a step that exceeds max_tokens
    remaining_tokens_budget = max(0, args.max_tokens - tokens_seen)
    max_allowed_steps_for_budget = remaining_tokens_budget // targets_per_optimizer_step
    max_steps_to_run = min(max_allowed_steps_for_budget, max_safe_session_steps)
    max_steps_to_run = max(0, max_steps_to_run)

    # Clean zero-update exit if budget is already reached or expired before taking any step
    elapsed_session = time.time() - t_session_start
    local_skip = (
        tokens_seen >= args.max_tokens
        or (tokens_seen + targets_per_optimizer_step > args.max_tokens)
        or max_steps_to_run <= 0
        or elapsed_session >= args.max_seconds
    )
    skip_training = check_coordinated_stop(local_skip, device)
    if skip_training and is_master:
        reason = "Budget already reached" if tokens_seen >= args.max_tokens else (
            "Next step would exceed cap" if (tokens_seen + targets_per_optimizer_step > args.max_tokens) else (
                "No safe steps remaining" if max_steps_to_run <= 0 else "Session duration expired"
            )
        )
        print(f"Zero-update exit trigger: {reason} (tokens_seen: {tokens_seen:,}, max_tokens: {args.max_tokens:,}, max_steps_to_run: {max_steps_to_run})")

    # 6. Training Loop Setup
    model.train()
    optimizer.zero_grad(set_to_none=True)

    seq_iter = [] if skip_training else dataset.get_sequence_iterator(start_shard=start_shard, start_offset=start_offset)
    batch_buffer = []

    global_step = start_step
    accum_step = 0
    accum_loss_tensor = torch.tensor(0.0, device=device)
    last_step_loss = 0.0

    tokens_trained_this_run = 0
    t_compilation_done = None
    t_warmup_done = None
    warmup_tokens_this_run = 0
    warmup_steps = min(5, max(1, total_steps // 10))
    ckpt_counter = 0

    current_shard_idx = start_shard
    current_seq_i = start_offset

    os.makedirs(args.output_dir, exist_ok=True)

    t_accum_start = time.time()

    try:
        for shard_idx, seq_i, tensor in seq_iter:
            # Check remaining budget before taking on new optimizer step
            if accum_step == 0 and (tokens_seen + targets_per_optimizer_step > args.max_tokens or global_step >= start_step + max_steps_to_run):
                if is_master:
                    reason = "Next step would exceed max_tokens cap" if (tokens_seen + targets_per_optimizer_step > args.max_tokens) else "Max safe steps reached"
                    print(f"Pre-step exit trigger: {reason} ({tokens_seen:,} targets)")
                break

            current_shard_idx = shard_idx
            current_seq_i = seq_i
            batch_buffer.append(tensor)

            if len(batch_buffer) == args.batch_size:
                # Static-shape batch: [batch_size, seq_len]
                input_ids = torch.stack(batch_buffer, dim=0).to(device)
                batch_buffer.clear()

                # Autoregressive next-token prediction
                labels = input_ids.clone()

                # Forward pass with finite loss check and BF16 autocast support
                autocast_device = "xla" if HAS_XLA else ("cuda" if torch.cuda.is_available() else "cpu")
                use_bf16 = getattr(args, "precision", "fp32") == "bf16"
                with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=use_bf16):
                    _, loss = model(input_ids=input_ids, labels=labels)
                del input_ids, labels
                scaled_loss = loss.float() / args.grad_accum_steps
                del loss
                scaled_loss.backward()
                accum_loss_tensor = accum_loss_tensor + scaled_loss.detach()
                del scaled_loss
                accum_step += 1

                # Optimizer step on gradient accumulation boundary
                if accum_step == args.grad_accum_steps:
                    # Coordinated all-rank finite loss check BEFORE gradient reduction
                    loss_finite = torch.isfinite(accum_loss_tensor).to(torch.int32)
                    if HAS_XLA:
                        loss_finite = xm.all_reduce(xm.REDUCE_MIN, loss_finite)
                        xm.mark_step()
                    if loss_finite.item() == 0:
                        raise FloatingPointError(
                            f"Non-finite loss detected across replicas at step {global_step} (rank {global_rank})"
                        )

                    # Single gradient reduction & clipping with finite gradient assertion
                    if HAS_XLA:
                        xm.reduce_gradients(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=args.max_grad_norm,
                        error_if_nonfinite=True,
                        foreach=False,
                    )

                    if HAS_XLA:
                        optimizer.step()
                        xm.mark_step(wait=True)
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Read scalar loss after device execution
                    last_step_loss = accum_loss_tensor.item()
                    accum_loss_tensor.zero_()

                    # Step timer captured AFTER device optimizer completion
                    step_duration = time.time() - t_accum_start
                    global_step += 1

                    valid_targets_this_step = targets_per_optimizer_step
                    tokens_seen += valid_targets_this_step
                    tokens_trained_this_run += valid_targets_this_step
                    accum_step = 0

                    if t_compilation_done is None:
                        t_compilation_done = time.time()
                        if is_master:
                            print(f"[XLA Warmup] First step compilation completed in {t_compilation_done - t_session_start:.2f}s")

                    if global_step == start_step + warmup_steps:
                        t_warmup_done = time.time()
                        warmup_tokens_this_run = tokens_trained_this_run

                    # Log progress (only log scalar loss on log interval to avoid device sync)
                    if global_step % args.log_interval == 0 and is_master:
                        tok_per_sec = valid_targets_this_step / max(step_duration, 1e-4)
                        lr_now = scheduler.get_last_lr()[0]
                        print(
                            f"Step {global_step:4d} | "
                            f"Targets: {tokens_seen:9,d} / {args.max_tokens:,d} | "
                            f"Loss: {last_step_loss:.4f} | "
                            f"LR: {lr_now:.2e} | "
                            f"Step Rate: {tok_per_sec:7.1f} tok/s"
                        )

                    # Persistent Checkpointing: computes deterministic cursor for EVERY rank
                    if (global_step - start_step) > 0 and (global_step - start_step) % args.save_interval == 0:
                        ckpt_counter += 1
                        ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step:05d}.pt")
                        seqs_consumed_this_run = (global_step - start_step) * args.batch_size * args.grad_accum_steps
                        seqs_consumed_so_far = initial_seqs_consumed_per_rank + seqs_consumed_this_run
                        all_rank_cursors = compute_all_rank_cursors(
                            all_shards=all_shards,
                            world_size=world_size,
                            seq_len=args.seq_len,
                            seqs_consumed_per_rank=seqs_consumed_so_far,
                            dtype=dataset.dtype,
                        )
                        data_state = {
                            "tokens_seen": tokens_seen,
                            "rank_cursors": all_rank_cursors,
                            "global_seqs_consumed_in_pack": seqs_consumed_so_far * world_size,
                        }
                        metrics = {"loss": last_step_loss}
                        save_checkpoint(
                            checkpoint_path=ckpt_path,
                            step=global_step,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            config=config,
                            data_state=data_state,
                            metrics=metrics,
                            identity_meta=expected_identity,
                        )
                        if is_master:
                            print(f"--> Saved persistent checkpoint: {ckpt_path}")
                            # Rotational checkpoint retention: keep only latest N intermediate checkpoints
                            keep_ckpts = getattr(args, "keep_checkpoints", 2)
                            if keep_ckpts > 0:
                                existing_ckpts = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint_step_*.pt")))
                                while len(existing_ckpts) > keep_ckpts:
                                    oldest = existing_ckpts.pop(0)
                                    try:
                                        os.remove(oldest)
                                        print(f"--> Rotated out old checkpoint: {oldest}")
                                    except Exception:
                                        pass

                    # Exit conditions check: coordinated across all ranks
                    elapsed_session = time.time() - t_session_start
                    local_stop = (
                        global_step >= start_step + max_steps_to_run
                        or tokens_seen >= args.max_tokens
                        or (tokens_seen + targets_per_optimizer_step > args.max_tokens)
                        or elapsed_session >= args.max_seconds
                    )
                    should_stop = check_coordinated_stop(local_stop, device)
                    if should_stop:
                        if is_master:
                            if tokens_seen >= args.max_tokens or (tokens_seen + targets_per_optimizer_step > args.max_tokens):
                                reason = "Token target budget reached"
                            elif global_step >= start_step + max_steps_to_run:
                                reason = "Maximum safe steps reached for data shards"
                            else:
                                reason = "Session time budget reached"
                            print(f"Exit trigger: {reason} ({tokens_seen:,} targets, {elapsed_session:.1f}s)")
                        break

                    t_accum_start = time.time()

    except Exception as e:
        if is_master:
            print(f"Error during training execution: {e}")
        raise e

    # Assert strict cap compliance
    assert tokens_seen <= args.max_tokens, f"Cap violation! Processed {tokens_seen:,} targets, exceeding max_tokens {args.max_tokens:,}"

    t_training_loop_end = time.time()

    # 7. Final Checkpoint and Metrics Persistence
    final_ckpt = os.path.join(args.output_dir, "tot_360m_calibration_final.pt")
    seqs_consumed_this_run = (global_step - start_step) * args.batch_size * args.grad_accum_steps
    seqs_consumed_so_far = initial_seqs_consumed_per_rank + seqs_consumed_this_run
    all_rank_cursors = compute_all_rank_cursors(
        all_shards=all_shards,
        world_size=world_size,
        seq_len=args.seq_len,
        seqs_consumed_per_rank=seqs_consumed_so_far,
        dtype=dataset.dtype,
    )
    data_state = {
        "tokens_seen": tokens_seen,
        "rank_cursors": all_rank_cursors,
        "global_seqs_consumed_in_pack": seqs_consumed_so_far * world_size,
    }
    save_checkpoint(
        checkpoint_path=final_ckpt,
        step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        data_state=data_state,
        metrics={"final_loss": last_step_loss},
        identity_meta=expected_identity,
    )

    # Calculate session end-to-end time including final checkpoint persistence
    total_elapsed = time.time() - t_session_start
    steady_state_targets = max(0, tokens_trained_this_run - warmup_tokens_this_run)
    steady_state_time = (t_training_loop_end - t_warmup_done) if t_warmup_done else (
        t_training_loop_end - (t_compilation_done if t_compilation_done else t_session_start)
    )
    steady_state_tok_s = (steady_state_targets / max(steady_state_time, 1e-4)) if steady_state_time > 0 else 0.0
    end_to_end_tok_s = (tokens_trained_this_run / max(total_elapsed, 1e-4)) if total_elapsed > 0 else 0.0

    if is_master:
        summary = {
            "status": "COMPLETED",
            "global_step": global_step,
            "total_valid_targets": tokens_seen,
            "tokens_trained_this_run": tokens_trained_this_run,
            "final_loss": round(last_step_loss, 4),
            "total_session_seconds": round(total_elapsed, 2),
            "compilation_seconds": round((t_compilation_done - t_session_start) if t_compilation_done else 0.0, 2),
            "steady_state_targets_per_sec": round(steady_state_tok_s, 1),
            "end_to_end_targets_per_sec": round(end_to_end_tok_s, 1),
            "checkpoints_saved": ckpt_counter + 1,
            "final_checkpoint_path": final_ckpt,
            "final_cursor": data_state["rank_cursors"],
            "config_fingerprint": get_config_fingerprint(config, optimizer.__class__.__name__),
        }
        summary_path = os.path.join(args.output_dir, "calibration_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n==========================================")
        print("Kaggle TPU Calibration Run Completed")
        print("==========================================")
        print(f"Total Valid Targets:  {tokens_seen:,}")
        print(f"Trained This Run:     {tokens_trained_this_run:,}")
        print(f"Total Session Time:   {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
        print(f"Steady-State Rate:    {steady_state_tok_s:.1f} targets/sec")
        print(f"End-to-End Rate:      {end_to_end_tok_s:.1f} targets/sec")
        print(f"Final Loss:           {last_step_loss:.4f}")
        print(f"Final Checkpoint:     {final_ckpt}")
        print(f"Summary Saved:        {summary_path}")
        print("==========================================\n")


def main():
    t_session_start = time.time()

    parser = argparse.ArgumentParser(description="Kaggle TPU Calibration & Pretraining Runner")
    parser.add_argument("--config", choices=["mini", "baseline"], default="baseline")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/checkpoints")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--allow-synthetic-shards", action="store_true", help="Explicitly allow synthetic shards for test fixtures only")
    parser.add_argument("--max-tokens", type=int, default=100_000_000, help="Max valid targets (default 100M)")
    parser.add_argument("--max-seconds", type=int, default=7000, help="Max session runtime seconds (< 2 hours)")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--keep-checkpoints", type=int, default=2, help="Number of rotational intermediate checkpoints to retain")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16" if HAS_XLA else "fp32", help="Precision mode (default bf16 on XLA, fp32 on CPU/CUDA)")
    parser.add_argument("--allow-tokenizer-mismatch", action="store_true", help="Allow running mini model against production tokenizer/vocab")
    parser.add_argument("--vocab-size-override", type=int, default=None, help="Explicitly override model vocab_size (e.g. 32000 for mini preflight on real shards)")
    parser.add_argument("--allow-cpu-baseline", action="store_true", help="Allow baseline model on CPU")
    parser.add_argument("--skip-checksum-verify", action="store_true", help="Skip checking shard SHA256 checksums at startup")
    parser.add_argument("--skip-identity-verify", action="store_true", help="Allow warm-starting across different shard packs or batch sizes while still checking config_fingerprint")
    parser.add_argument("--reset-data-cursor", action="store_true", help="Reset shard/sequence cursor to (0, 0) when starting a new bulk shard pack from a prior checkpoint")
    args = parser.parse_args()

    if HAS_XLA:
        xmp.spawn(train_worker, args=(args, t_session_start), nprocs=None)
    else:
        train_worker(0, args, t_session_start)


if __name__ == "__main__":
    main()
