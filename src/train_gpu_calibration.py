"""Bounded CUDA candidate; CPU is admitted only for explicit miniature fixtures.

This file does not change the submitted TPU recipe. Run with torchrun for DDP.
Real CUDA admission, FP16 scaler recovery and full-size speed remain hardware gates.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tot_360m.config import Tot360MConfig, TotMiniConfig
from tot_360m.src.model import TotForCausalLM

TOKENIZER_SHA = "1ebc8d7aab0e6d2d12a6b4da34fcf6d84f136dcad0c6c1a66e600a3d48b9dba9"
MODEL_SHA = "9961432ef823ae95b5d22d62d41e79c0d5afff7ea208a1013197f525e87af674"
MANIFEST_SHA = "be9bf24c7ce842dec675c516586e1d1dfb2ef3364eca4efbccd717cdb978c41c"
EXPECTED_PARAMETERS = 345359296


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def architecture_fingerprint(config):
    """Architecture only; runtime is recorded separately, never silently remapped."""
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def admit_device(config_name, device_name, fixture, precision, cuda_available,
                 cuda_devices, world_size, bf16_supported):
    if device_name == "cpu":
        if config_name != "mini" or not fixture:
            raise RuntimeError("CPU execution is permitted only for --config mini --fixture tests")
    elif device_name != "cuda" or not cuda_available or cuda_devices < world_size:
        raise RuntimeError("Genuine CUDA devices for every requested rank are required")
    if precision == "bf16" and device_name == "cuda" and not bf16_supported:
        raise RuntimeError("This CUDA device lacks native BF16; use FP16 with GradScaler")
    if fixture and config_name != "mini":
        raise RuntimeError("Fixtures cannot train the baseline")


def validate_manifest(data_dir, config, seq_len, fixture=False, allow_bulk_manifest=False, skip_checksum_verify=False):
    root = Path(data_dir).resolve()
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not fixture and not allow_bulk_manifest and sha256(path) != MANIFEST_SHA:
        raise ValueError("Calibration manifest differs from the reviewed snapshot")
    if not fixture and manifest.get("tokenizer_json_sha256") != TOKENIZER_SHA:
        raise ValueError("Tokenizer identity mismatch")
    if manifest.get("dtype") != "uint16" or manifest.get("seq_len") != seq_len:
        raise ValueError("Expected uint16 shards at the exact manifest sequence length")
    if manifest.get("vocab_size") != config.vocab_size or not manifest.get("shards"):
        raise ValueError("Invalid vocabulary or empty shard list")
    shards = []
    names = set()
    for item in sorted(manifest["shards"], key=lambda s: s["name"]):
        p = (root / item["name"]).resolve()
        if not p.is_relative_to(root) or item["name"] in names:
            raise ValueError("Unsafe or duplicate shard path")
        names.add(item["name"])
        size = p.stat().st_size
        if not size or size % (seq_len * 2) or (not skip_checksum_verify and sha256(p) != item.get("sha256")):
            raise ValueError(f"Shard integrity failure: {item['name']}")
        n = size // (seq_len * 2)
        if n != item.get("seq_count"):
            raise ValueError("Shard sequence count differs from actual size")
        if skip_checksum_verify:
            with open(p, "rb") as f:
                first_seq = f.read(seq_len * 2)
            if first_seq and int(np.frombuffer(first_seq, dtype="<u2").max()) >= config.vocab_size:
                raise ValueError("Out-of-vocabulary token in shard")
        else:
            # Check the entire calibration corpus, not only its first sequence.
            data = np.memmap(p, mode="r", dtype="<u2")
            if int(data.max()) >= config.vocab_size:
                raise ValueError("Out-of-vocabulary token in shard")
            del data
        shards.append({"name": item["name"], "path": p, "count": n, "sha256": item["sha256"]})
    return manifest, shards, sha256(path)


class GlobalCursor:
    """One canonical sequence stream, split into disjoint rank microbatches.

    Cursor advances only after a successful optimizer update. A failed FP16 step
    reuses the exact descriptors; incomplete final groups remain available.
    Supports exact unconsumed tail resolution when resuming mid-pack from a
    multi-rank TPU checkpoint (`rank_cursors`).
    """
    def __init__(self, shards, seq_len, position=0):
        self.shards = shards
        self.seq_len = seq_len
        self.position = position
        self.total = sum(s["count"] for s in shards)
        self.consumed_per_shard = [0] * len(shards)
        self.base_position = 0
        if not 0 <= position <= self.total:
            raise ValueError("Checkpoint sequence cursor is outside the corpus")

    def set_tpu_rank_cursors(self, rank_cursors):
        if not rank_cursors or not self.shards or self.position <= 0:
            self.consumed_per_shard = [0] * len(self.shards)
            self.base_position = 0
            return
        num_tpu_ranks = len(rank_cursors)
        if num_tpu_ranks <= 1 or not all(str(r) in rank_cursors for r in range(num_tpu_ranks)):
            self.consumed_per_shard = [0] * len(self.shards)
            self.base_position = 0
            return
        consumed = []
        for s_idx, shard in enumerate(self.shards):
            r = s_idx % num_tpu_ranks
            k = s_idx // num_tpu_ranks
            cur = rank_cursors.get(str(r), {})
            r_shard_idx = int(cur.get("shard_idx", 0))
            r_seq_offset = int(cur.get("seq_offset", 0))
            if k < r_shard_idx:
                consumed.append(shard["count"])
            elif k == r_shard_idx:
                consumed.append(min(max(0, r_seq_offset), shard["count"]))
            else:
                consumed.append(0)
        if sum(consumed) == self.position:
            self.consumed_per_shard = consumed
            self.base_position = self.position

    def plan(self, count):
        if count <= 0 or self.position + count > self.total:
            raise ValueError("Insufficient unique sequences; no replay or padding")
        descriptors = []
        if self.base_position > 0 and self.position >= self.base_position:
            start = self.position - self.base_position
            for shard, base_off in zip(self.shards, self.consumed_per_shard):
                avail = shard["count"] - base_off
                if avail <= 0:
                    continue
                if start >= avail:
                    start -= avail
                    continue
                take = min(count - len(descriptors), avail - start)
                descriptors.extend((shard["name"], base_off + start + i) for i in range(take))
                start = 0
                if len(descriptors) == count:
                    break
            return descriptors
        start = self.position
        for shard in self.shards:
            if start >= shard["count"]:
                start -= shard["count"]
                continue
            take = min(count - len(descriptors), shard["count"] - start)
            descriptors.extend((shard["name"], start + i) for i in range(take))
            start = 0
            if len(descriptors) == count:
                break
        return descriptors

    def read(self, descriptors):
        paths = {s["name"]: s["path"] for s in self.shards}
        result = []
        for name, offset in descriptors:
            with open(paths[name], "rb") as f:
                f.seek(offset * self.seq_len * 2)
                buf = f.read(self.seq_len * 2)
            if len(buf) != self.seq_len * 2:
                raise ValueError("Shard changed or sequence read was incomplete")
            ids = np.frombuffer(buf, dtype="<u2").astype(np.int64)
            result.append(torch.from_numpy(ids))
        return torch.stack(result)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def capture_rng(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).cpu()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        if "cuda" not in state:
            raise ValueError("CUDA rank RNG state is missing")
        torch.cuda.set_rng_state(state["cuda"], device)


def write_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def load_payload(path, identity, require_scaler, allow_cross_backend=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if allow_cross_backend and payload.get("version") == 2 and "data_state" in payload:
        data_state = payload["data_state"]
        tokens_seen = int(data_state.get("tokens_seen", 0))
        if "global_seqs_consumed_in_pack" in data_state:
            cursor_pos = int(data_state["global_seqs_consumed_in_pack"])
        else:
            cursor_pos = tokens_seen // max(1, identity["seq_len"] - 1)
        return {
            "format": "tot-gpu-candidate-v1",
            "from_cross_backend": True,
            "identity": identity,
            "step": int(payload.get("step", 0)),
            "tokens_seen": tokens_seen,
            "cursor": cursor_pos,
            "rank_cursors": data_state.get("rank_cursors"),
            "model_state": payload.pop("model_state"),
            "optimizer_state": payload.pop("optimizer_state", None),
            "scheduler_state": payload.pop("scheduler_state", None),
            "scaler_state": payload.pop("scaler_state", None) or {},
            "rank_states": [payload.get("rng_states", {})] * identity["world_size"],
        }
    if payload.get("format") != "tot-gpu-candidate-v1" or (not allow_cross_backend and payload.get("identity") != identity):
        raise ValueError("GPU checkpoint identity mismatch; cross-backend import is not yet admitted")
    for key in ("model_state", "optimizer_state", "scheduler_state", "rank_states", "cursor", "tokens_seen", "step"):
        if key not in payload or payload[key] is None:
            raise ValueError(f"Required checkpoint state is missing: {key}")
    if require_scaler and not payload.get("scaler_state") and not allow_cross_backend:
        raise ValueError("FP16 GradScaler state is missing; unsafe resume rejected")
    return payload


def recursive_equal(a, b, atol=0.0, path="state"):
    """Strict shape/key/type comparison, including integer counters and empty tensors."""
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"Tensor mismatch at {path}")
        if a.is_floating_point():
            if not torch.allclose(a, b, atol=atol, rtol=0.0):
                raise AssertionError(f"Numerical mismatch at {path}")
        elif not torch.equal(a, b):
            raise AssertionError(f"Integer mismatch at {path}")
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or not np.array_equal(a, b):
            raise AssertionError(f"Array mismatch at {path}")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise AssertionError(f"Keys mismatch at {path}")
        for key in a:
            recursive_equal(a[key], b[key], atol, f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(f"Sequence mismatch at {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            recursive_equal(x, y, atol, f"{path}[{i}]")
    elif type(a) is not type(b) or a != b:
        raise AssertionError(f"Value mismatch at {path}")


def rank_value(value, device, world, op=dist.ReduceOp.MIN):
    t = torch.tensor(int(value), device=device, dtype=torch.int32)
    if world > 1:
        dist.all_reduce(t, op=op)
    return bool(t.item())


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not 0 <= rank < world or world not in (1, 2):
        raise ValueError("GPU candidate supports one or two explicit ranks")
    device = torch.device(args.device, local_rank) if args.device == "cuda" else torch.device("cpu")
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    admit_device(args.config, args.device, args.fixture, args.precision,
                 torch.cuda.is_available(), torch.cuda.device_count(), world,
                 torch.cuda.is_bf16_supported(including_emulation=False) if torch.cuda.is_available() else False)
    if args.config == "baseline" and sha256(Path(__file__).with_name("model.py")) != MODEL_SHA:
        raise ValueError("Baseline model source differs from the independently reviewed architecture")
    config = TotMiniConfig() if args.config == "mini" else Tot360MConfig()
    if args.config == "mini" and not args.fixture:
        config.vocab_size = 32000
        config.max_position_embeddings = Tot360MConfig().max_position_embeddings
    if args.seq_len > config.max_position_embeddings or args.seq_len < 2:
        raise ValueError("Sequence length is outside the frozen context bounds")
    manifest, shards, manifest_hash = validate_manifest(
        args.data_dir, config, args.seq_len, args.fixture,
        getattr(args, "allow_bulk_manifest", False),
        getattr(args, "skip_checksum_verify", False),
    )
    if world > 1:
        from datetime import timedelta
        options = {}
        if args.fixture_rendezvous:
            if device.type != "cpu" or not args.fixture or args.config != "mini":
                raise ValueError("File rendezvous is only for local miniature CPU fixtures")
            options = {"init_method": Path(args.fixture_rendezvous).resolve().as_uri(), "rank": rank, "world_size": world}
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=180), **options)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    model = TotForCausalLM(config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if args.config == "baseline" and parameter_count != EXPECTED_PARAMETERS:
        raise ValueError("Frozen baseline parameter count changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                                  foreach=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_steps, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=args.precision == "fp16", init_scale=128.0)
    identity = {"architecture": architecture_fingerprint(config), "config": asdict(config),
                "manifest_sha256": manifest_hash, "tokenizer_sha256": manifest.get("tokenizer_json_sha256"),
                "world_size": world, "batch_size": args.batch_size, "accumulation": args.grad_accum_steps,
                "seq_len": args.seq_len, "precision": args.precision, "device": device.type,
                "torch_release": torch.__version__.split("+")[0], "schedule_steps": args.schedule_steps,
                "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1},
                "seed": args.seed, "clip_norm": 1.0}
    step = tokens_seen = 0
    cursor = GlobalCursor(shards, args.seq_len)
    if args.resume:
        allow_cross = getattr(args, "allow_cross_backend", False)
        allow_bulk = getattr(args, "allow_bulk_manifest", False)
        reset_cur = getattr(args, "reset_data_cursor", False)
        payload = load_payload(args.resume, identity, scaler.is_enabled(), allow_cross_backend=allow_cross)
        if not allow_cross:
            if len(payload["rank_states"]) != world:
                raise ValueError("Checkpoint rank/cursor/token accounting is inconsistent")
            expected_pack_tokens = payload["cursor"] * (args.seq_len - 1)
            if allow_bulk or reset_cur:
                if payload["tokens_seen"] < expected_pack_tokens:
                    raise ValueError("Checkpoint rank/cursor/token accounting is inconsistent")
            elif payload["tokens_seen"] != expected_pack_tokens:
                raise ValueError("Checkpoint rank/cursor/token accounting is inconsistent")
        model.load_state_dict(payload.pop("model_state"))
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        opt_state = payload.pop("optimizer_state", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            del opt_state
        sched_state = payload.pop("scheduler_state", None)
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
            del sched_state
        scaler_state = payload.pop("scaler_state", None)
        if scaler_state:
            scaler.load_state_dict(scaler_state)
            del scaler_state
        if not payload.get("from_cross_backend") and rank < len(payload["rank_states"]):
            restore_rng(payload["rank_states"][rank], device)
        cursor.position = 0 if reset_cur else payload["cursor"]
        if not 0 <= cursor.position <= cursor.total:
            raise ValueError("Resume cursor outside immutable corpus")
        if not reset_cur:
            if payload.get("rank_cursors"):
                cursor.set_tpu_rank_cursors(payload.get("rank_cursors"))
            elif payload.get("tpu_consumed_per_shard") and payload.get("tpu_base_position"):
                cursor.consumed_per_shard = list(payload["tpu_consumed_per_shard"])
                cursor.base_position = int(payload["tpu_base_position"])
        step, tokens_seen = payload["step"], payload["tokens_seen"]
        del payload
        gc.collect()
    raw_model = model
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    broadcast_buffers=False, gradient_as_bucket_view=True)
    model.train()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    global_batch = world * args.batch_size * args.grad_accum_steps
    step_targets = global_batch * (args.seq_len - 1)
    if (getattr(args, "allow_bulk_manifest", False) or getattr(args, "allow_cross_backend", False)) and args.schedule_steps <= step:
        desired_total_steps = step + math.ceil(max(0, args.max_tokens - tokens_seen) / max(step_targets, 1))
        args.schedule_steps = max(desired_total_steps, step + args.max_steps, 100)
    if getattr(scheduler, "T_max", 0) < args.schedule_steps:
        scheduler.T_max = args.schedule_steps
    initial_step, initial_tokens = step, tokens_seen
    durations = []
    losses = []
    consumed_ids = []
    overflows = 0
    deadline = started + args.max_seconds
    if os.environ.get("TOT_SESSION_DEADLINE_UNIX"):
        deadline = min(deadline, time.monotonic() + float(os.environ["TOT_SESSION_DEADLINE_UNIX"]) - time.time())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def save(path):
        sync()
        states = [None] * world
        local_state = capture_rng(device)
        if world > 1:
            dist.all_gather_object(states, local_state)
        else:
            states[0] = local_state
        if rank == 0:
            ckpt_dict = {"format": "tot-gpu-candidate-v1", "identity": identity,
                "step": step, "tokens_seen": tokens_seen, "cursor": cursor.position,
                "model_state": cpu_tree(raw_model.state_dict()), "optimizer_state": cpu_tree(optimizer.state_dict()),
                "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(), "rank_states": states,
                "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda}, "parameter_count": parameter_count}
            if cursor.base_position > 0:
                ckpt_dict["tpu_consumed_per_shard"] = list(cursor.consumed_per_shard)
                ckpt_dict["tpu_base_position"] = int(cursor.base_position)
            write_atomic(path, ckpt_dict)
            del ckpt_dict
            gc.collect()
        if world > 1:
            dist.barrier()

    reason = "unknown"
    log_interval = max(1, int(getattr(args, "log_interval", 1)))
    try:
        while True:
            can_start = (time.monotonic() + args.save_margin < deadline
                         and step - initial_step < args.max_steps and step < args.schedule_steps
                         and tokens_seen + step_targets <= args.max_tokens
                         and cursor.position + global_batch <= cursor.total)
            if not rank_value(can_start, device, world):
                reason = "time, step, token limit or unique corpus boundary"
                break
            descriptors = cursor.plan(global_batch)
            # One deterministic global step split across ranks and accumulation.
            batches = []
            for micro in range(args.grad_accum_steps):
                start = (micro * world + rank) * args.batch_size
                batch = cursor.read(descriptors[start:start + args.batch_size])
                if device.type == "cuda":
                    batch = batch.pin_memory()
                batches.append(batch.to(device, non_blocking=device.type == "cuda"))
            sync()
            t0 = time.monotonic()
            for attempt in range(8):
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=device)
                for micro, batch in enumerate(batches):
                    no_sync = model.no_sync() if world > 1 and micro + 1 < len(batches) else contextlib.nullcontext()
                    with no_sync:
                        dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.precision != "fp32"):
                            _, loss = model(input_ids=batch, labels=batch)
                        scaler.scale(loss.float() / args.grad_accum_steps).backward()
                        loss_sum += loss.detach().float() / args.grad_accum_steps
                if not rank_value(torch.isfinite(loss_sum).item(), device, world):
                    raise FloatingPointError("Non-finite loss; no update or cursor advancement")
                scaler.unscale_(optimizer)
                # One host synchronization, rather than one per parameter tensor.
                finite = torch.stack([torch.isfinite(p.grad).all() for p in raw_model.parameters() if p.grad is not None]).all().item()
                if not rank_value(finite, device, world):
                    if not scaler.is_enabled():
                        raise FloatingPointError("Non-finite FP32/BF16 gradients")
                    scaler.update(new_scale=scaler.get_scale() * 0.5)
                    overflows += 1
                    continue  # retry the SAME data, without scheduler/token/cursor advancement
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                sync()
                elapsed = time.monotonic() - t0
                if world > 1:
                    t = torch.tensor(elapsed, dtype=torch.float64, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    elapsed = t.item()
                    dist.all_reduce(loss_sum)
                    loss_sum /= world
                durations.append(elapsed)
                losses.append(loss_sum.item())
                if args.record_batch_ids:
                    consumed_ids.extend(descriptors)
                step += 1
                tokens_seen += step_targets
                cursor.position += global_batch
                break
            else:
                raise FloatingPointError("Eight FP16 overflow retries exhausted on the same batch")
            if rank == 0 and (step % log_interval == 0 or step - initial_step == 1):
                print(json.dumps({"step": step, "valid_targets": tokens_seen, "loss": losses[-1],
                                  "targets_per_second": step_targets / durations[-1], "scale": scaler.get_scale()}), flush=True)
            if args.save_interval and step % args.save_interval == 0:
                save(output / "latest.pt")
        save(output / "final.pt")
        if rank == 0:
            warm = min(args.warmup_steps, len(durations))
            steady_seconds = sum(durations[warm:])
            total_seconds = time.monotonic() - started
            summary = {"status": "BOUNDED_COMPLETE", "termination_reason": reason, "device": str(device),
                "world_size": world, "precision": args.precision, "parameter_count": parameter_count,
                "steps_this_run": step - initial_step, "global_step": step, "total_valid_targets": tokens_seen,
                "targets_this_run": tokens_seen - initial_tokens, "global_batch_sequences": global_batch,
                "step_seconds": durations, "losses": losses, "warmup_steps": warm, "overflow_retries": overflows,
                "steady_state_targets_per_second": (len(durations) - warm) * step_targets / steady_seconds if steady_seconds else None,
                "end_to_end_targets_per_second": (tokens_seen - initial_tokens) / total_seconds,
                "total_seconds": total_seconds, "cursor": cursor.position, "remaining_unique_sequences": cursor.total - cursor.position,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "checkpoint_sha256": sha256(output / "final.pt"), "identity": identity,
                "batch_ids": consumed_ids if args.record_batch_ids else None,
                "hardware_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU mini fixture"}
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:
        # Never serialize partially updated or non-finite weights as a good checkpoint.
        if rank == 0:
            (output / "failure.json").write_text(json.dumps({"error": type(exc).__name__, "message": str(exc),
                "last_completed_step": step, "tokens_seen": tokens_seen, "latest_checkpoint": str(output / "latest.pt")}), encoding="utf-8")
        raise
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=["baseline", "mini"], default="baseline")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--fixture-rendezvous", help="File rendezvous for local two-rank CPU fixture tests only")
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=100000000)
    p.add_argument("--max-seconds", type=int, default=1200)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--schedule-steps", type=int, default=6109)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--save-margin", type=int, default=120)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--record-batch-ids", action="store_true")
    p.add_argument("--allow-cross-backend", action="store_true", help="Allow resuming from a TPU v2 checkpoint")
    p.add_argument("--allow-bulk-manifest", action="store_true", help="Allow bulk uint16 manifests and full session bounds")
    p.add_argument("--reset-data-cursor", action="store_true", help="Reset sequence cursor to 0 when starting a new bulk shard pack")
    p.add_argument("--skip-checksum-verify", action="store_true", help="Skip full-file SHA256/memmap scan on pre-verified bulk shards")
    args = p.parse_args()
    for name in ("batch_size", "grad_accum_steps", "max_steps", "schedule_steps", "max_tokens", "max_seconds"):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    max_tok_cap = 10_000_000_000 if args.allow_bulk_manifest else 100_000_000
    max_sec_cap = 43_200 if args.allow_bulk_manifest else 7_000
    if args.max_tokens > max_tok_cap or args.max_seconds > max_sec_cap or args.save_margin < 0 or args.save_interval < 0 or args.warmup_steps < 0:
        p.error("Candidate exceeds smoke bounds or has a negative safety/timing setting")
    return args


if __name__ == "__main__":
    run(parse_args())
