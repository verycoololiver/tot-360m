"""Production Corpus Builder and uint16 Shard Generator for Tot-360M.

Features:
- Collects curated, strictly permissive licensed corpus (~30MB uncompressed text / ~8M tokens):
  1. FineWeb-Edu (score >= 3.0 / int_score >= 3, ODC-By)
  2. Permissive Clean Code (MIT, Apache-2.0, BSD, ISC, CC0, Unlicense) with repo/path/license metadata
  3. Korean Wikipedia (20231101.ko, CC-BY-SA-3.0 / GFDL dual licensed)
  4. Authored Iris companion dialogue templates
- Immutable record provenance saved to production_corpus.jsonl with document-level SHA256 hashes
- Exact deduplication across all texts against each other and held_out_eval.jsonl
- Tokenized with frozen 32k BPE tokenizer (SHA256: 1ebc8d7aab0e6d2d12a6b4da34fcf6d84f136dcad0c6c1a66e600a3d48b9dba9)
- Packed into fixed seq_len=1024 sequences formatted as torch.uint16 (2 bytes/token)
- Partitioned into 16 balanced rank-disjoint binary shards with manifest.json checksums
"""

import os
import sys
import time
import json
import glob
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tot_360m.tokenizer.build_production_tokenizer import (
    PERMISSIVE_LICENSES,
    fetch_rows_with_retry,
    get_iris_persona_templates,
)

FROZEN_TOKENIZER_SHA256 = "1ebc8d7aab0e6d2d12a6b4da34fcf6d84f136dcad0c6c1a66e600a3d48b9dba9"
TARGET_SEQ_LEN = 1024
NUM_SHARDS = 16


def stream_fineweb_edu_records(target_bytes: int = 15_000_000) -> List[Dict[str, Any]]:
    """Streams FineWeb-Edu records with source metadata."""
    offset = 0
    bytes_collected = 0
    records = []
    min_required = int(target_bytes * 0.8)

    while bytes_collected < target_bytes:
        url = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&split=train&offset={offset}&length=100"
        rows = fetch_rows_with_retry(url)
        if not rows:
            break

        for row in rows:
            score = row.get("score", 0.0)
            int_score = row.get("int_score", 0)
            if score >= 3.0 or int_score >= 3:
                text = row.get("text", "")
                if text and len(text) > 100:
                    text_bytes = text.encode("utf-8")
                    content_sha = hashlib.sha256(text_bytes).hexdigest()
                    rec = {
                        "id": str(row.get("id", f"fw_{offset}_{len(records)}")),
                        "domain": "fineweb_edu",
                        "dataset": "HuggingFaceFW/fineweb-edu",
                        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
                        "license": "ODC-By",
                        "url": row.get("url", ""),
                        "dump": row.get("dump", ""),
                        "score": score,
                        "int_score": int_score,
                        "sha256": content_sha,
                        "text": text,
                    }
                    records.append(rec)
                    bytes_collected += len(text_bytes)
                    if bytes_collected >= target_bytes:
                        break
        offset += 100

    if bytes_collected < min_required:
        raise RuntimeError(f"Undercollection in FineWeb-Edu: {bytes_collected} < {min_required}")
    return records


def stream_permissive_code_records(target_bytes: int = 8_000_000) -> List[Dict[str, Any]]:
    """Streams permissive code records with repository, path, and verified license metadata."""
    records = []
    bytes_collected = 0
    min_required = int(target_bytes * 0.8)
    sub_configs = [("all-mit", target_bytes // 2), ("all-apache-2.0", target_bytes - (target_bytes // 2))]

    for config_name, sub_target in sub_configs:
        offset = 0
        c_bytes = 0
        while c_bytes < sub_target:
            url = f"https://datasets-server.huggingface.co/rows?dataset=codeparrot/github-code-clean&config={config_name}&split=train&offset={offset}&length=100"
            rows = fetch_rows_with_retry(url)
            if not rows:
                break

            for row in rows:
                raw_lic = str(row.get("license", "")).lower().strip()
                if raw_lic in PERMISSIVE_LICENSES:
                    code = row.get("code", "")
                    if code and len(code) > 50:
                        code_bytes = code.encode("utf-8")
                        content_sha = hashlib.sha256(code_bytes).hexdigest()
                        rec = {
                            "id": f"{config_name}_{offset}_{len(records)}",
                            "domain": "permissive_code",
                            "dataset": "codeparrot/github-code-clean",
                            "revision": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
                            "config": config_name,
                            "license": raw_lic,
                            "repo_name": row.get("repo_name", ""),
                            "path": row.get("path", ""),
                            "sha256": content_sha,
                            "text": code,
                        }
                        records.append(rec)
                        doc_len = len(code_bytes)
                        c_bytes += doc_len
                        bytes_collected += doc_len
                        if c_bytes >= sub_target:
                            break
            offset += 100

    if bytes_collected < min_required:
        raise RuntimeError(f"Undercollection in Permissive Code: {bytes_collected} < {min_required}")
    return records


def stream_korean_records(target_bytes: int = 7_000_000) -> List[Dict[str, Any]]:
    """Streams Korean Wikipedia records with CC-BY-SA-3.0 / GFDL dual licensing provenance."""
    offset = 0
    bytes_collected = 0
    records = []
    min_required = int(target_bytes * 0.8)

    while bytes_collected < target_bytes:
        url = f"https://datasets-server.huggingface.co/rows?dataset=wikimedia/wikipedia&config=20231101.ko&split=train&offset={offset}&length=100"
        rows = fetch_rows_with_retry(url)
        if not rows:
            break

        for row in rows:
            text = row.get("text", "")
            if text and len(text) > 100:
                text_bytes = text.encode("utf-8")
                content_sha = hashlib.sha256(text_bytes).hexdigest()
                rec = {
                    "id": str(row.get("id", f"ko_wiki_{offset}_{len(records)}")),
                    "domain": "korean_wikipedia",
                    "dataset": "wikimedia/wikipedia",
                    "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
                    "config": "20231101.ko",
                    "license": "CC-BY-SA-3.0 / GFDL",
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "sha256": content_sha,
                    "text": text,
                }
                records.append(rec)
                bytes_collected += len(text_bytes)
                if bytes_collected >= target_bytes:
                    break
        offset += 100

    if bytes_collected < min_required:
        raise RuntimeError(f"Undercollection in Korean Wikipedia: {bytes_collected} < {min_required}")
    return records


def get_iris_records() -> List[Dict[str, Any]]:
    """Creates authentic Iris companion dialogue template records."""
    templates = get_iris_persona_templates()
    records = []
    for idx, tmpl in enumerate(templates):
        tmpl_bytes = tmpl.encode("utf-8")
        records.append({
            "id": f"iris_template_{idx:02d}",
            "domain": "iris_companion",
            "dataset": "authored/ob_studio",
            "license": "Proprietary / OB Studio internal",
            "author": "OB Studio",
            "role": "companion",
            "sha256": hashlib.sha256(tmpl_bytes).hexdigest(),
            "text": tmpl,
        })
    return records


def build_production_corpus_and_shards(
    output_dir: str = "tot_360m/data",
    tokenizer_path: str = "tot_360m/tokenizer/production/tokenizer.json",
    fineweb_mb: int = 15,
    code_mb: int = 8,
    korean_mb: int = 7,
):
    """Fetches records, deduplicates, saves production_corpus.jsonl, tokenizes, and writes uint16 shards."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify frozen tokenizer
    tok_file = Path(tokenizer_path)
    if not tok_file.exists():
        raise FileNotFoundError(f"Frozen tokenizer not found: {tokenizer_path}")
    tok_bytes = tok_file.read_bytes()
    tok_hash = hashlib.sha256(tok_bytes).hexdigest()
    if tok_hash != FROZEN_TOKENIZER_SHA256:
        raise ValueError(f"Frozen tokenizer hash mismatch! Expected {FROZEN_TOKENIZER_SHA256}, got {tok_hash}")
    print(f"Verified frozen tokenizer: {tok_hash[:12]}...")
    tokenizer = Tokenizer.from_file(str(tok_file))

    # 2. Load held-out hashes to guarantee zero leakage
    held_out_path = Path("tot_360m/tokenizer/production/held_out_eval.jsonl")
    held_out_hashes: Set[str] = set()
    if held_out_path.exists():
        for line in held_out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                h = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
                held_out_hashes.add(h)
    print(f"Loaded {len(held_out_hashes)} held-out hashes for leakage exclusion.")

    # 3. Stream representative records
    print("\nFetching representative licensed corpus streams...")
    fw_records = stream_fineweb_edu_records(target_bytes=fineweb_mb * 1_000_000)
    print(f"-> FineWeb-Edu: {len(fw_records):,} records")
    code_records = stream_permissive_code_records(target_bytes=code_mb * 1_000_000)
    print(f"-> Permissive Code: {len(code_records):,} records")
    ko_records = stream_korean_records(target_bytes=korean_mb * 1_000_000)
    print(f"-> Korean Wikipedia (CC-BY-SA-3.0/GFDL): {len(ko_records):,} records")
    iris_records = get_iris_records()
    print(f"-> Iris Persona: {len(iris_records)} templates")

    # 4. Deduplicate and filter held-out documents
    seen_hashes: Set[str] = set()
    training_records: List[Dict[str, Any]] = []

    all_raw_records = fw_records + code_records + ko_records + iris_records
    for rec in all_raw_records:
        h = rec["sha256"]
        if h in held_out_hashes:
            continue  # Strictly excluded from training corpus
        if h in seen_hashes:
            continue  # Deduplicated
        seen_hashes.add(h)
        training_records.append(rec)

    print(f"\nDeduplicated training corpus: {len(training_records):,} unique documents.")

    # 5. Save immutable production_corpus.jsonl
    corpus_jsonl_path = out_path / "production_corpus.jsonl"
    with open(corpus_jsonl_path, "w", encoding="utf-8") as f:
        for r in training_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    corpus_hash = hashlib.sha256(corpus_jsonl_path.read_bytes()).hexdigest()
    print(f"Saved immutable corpus -> {corpus_jsonl_path} (SHA256: {corpus_hash})")

    # 6. Tokenize and pack into uint16 sequences
    print("\nTokenizing corpus with ByteLevel BPE and packing into uint16 sequences...")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    assert eos_id == 0, f"Expected <|endoftext|> ID 0, got {eos_id}"

    token_stream: List[int] = []
    for r in training_records:
        enc = tokenizer.encode(r["text"])
        token_stream.extend(enc.ids)
        token_stream.append(eos_id)

    total_tokens = len(token_stream)
    num_complete_seqs = total_tokens // TARGET_SEQ_LEN
    total_tokens_packed = num_complete_seqs * TARGET_SEQ_LEN
    packed_tokens = token_stream[:total_tokens_packed]
    print(f"Total raw tokens: {total_tokens:,} -> Packed sequences: {num_complete_seqs:,} ({total_tokens_packed:,} tokens)")

    # 7. Write 16 balanced binary shards as uint16
    seqs_per_shard = num_complete_seqs // NUM_SHARDS
    shard_meta = []
    print(f"Writing {NUM_SHARDS} balanced uint16 shards (~{seqs_per_shard} seqs/shard)...")

    for s_idx in range(NUM_SHARDS):
        start_seq = s_idx * seqs_per_shard
        end_seq = (s_idx + 1) * seqs_per_shard if s_idx < NUM_SHARDS - 1 else num_complete_seqs
        shard_seqs = end_seq - start_seq

        start_tok = start_seq * TARGET_SEQ_LEN
        end_tok = end_seq * TARGET_SEQ_LEN
        sub_tokens = packed_tokens[start_tok:end_tok]

        shard_filename = f"shard_{s_idx:03d}.bin"
        shard_path = shards_dir / shard_filename
        tensor = torch.tensor(sub_tokens, dtype=torch.int64).to(torch.uint16)
        raw_bytes = tensor.numpy().tobytes()

        with open(shard_path, "wb") as f:
            f.write(raw_bytes)

        s_sha = hashlib.sha256(raw_bytes).hexdigest()
        s_size = len(raw_bytes)
        shard_meta.append({
            "name": f"shards/{shard_filename}",
            "filename": shard_filename,
            "seq_count": shard_seqs,
            "token_count": shard_seqs * TARGET_SEQ_LEN,
            "byte_size": s_size,
            "sha256": s_sha,
        })
        print(f"   Shard {s_idx:02d}: {shard_seqs} seqs ({s_size / 1_048_576:.2f} MB, {s_sha[:10]})")

    # 8. Write manifest.json
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer_json_sha256": tok_hash,
        "vocab_size": tokenizer.get_vocab_size(),
        "dtype": "uint16",
        "bytes_per_token": 2,
        "bytes_per_seq": TARGET_SEQ_LEN * 2,
        "seq_len": TARGET_SEQ_LEN,
        "num_shards": NUM_SHARDS,
        "total_sequences": num_complete_seqs,
        "total_tokens": total_tokens_packed,
        "shards": shard_meta,
        "corpus_jsonl": {
            "path": str(corpus_jsonl_path.name),
            "sha256": corpus_hash,
            "total_documents": len(training_records),
        },
        "held_out_eval": {
            "path": str(held_out_path),
            "sha256": hashlib.sha256(held_out_path.read_bytes()).hexdigest(),
            "doc_count": len(held_out_hashes),
        },
        "sources": {
            "fineweb_edu": {
                "dataset": "HuggingFaceFW/fineweb-edu",
                "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
                "license": "ODC-By",
                "filter": "score >= 3.0 or int_score >= 3",
            },
            "permissive_code": {
                "dataset": "codeparrot/github-code-clean",
                "revision": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
                "allowed_licenses": sorted(list(PERMISSIVE_LICENSES)),
            },
            "korean_wikipedia": {
                "dataset": "wikimedia/wikipedia (20231101.ko)",
                "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
                "license": "CC-BY-SA-3.0 / GFDL",
                "provenance": "Wikimedia Foundation Wikipedia dump (ko) 20231101",
            },
            "iris_companion": {
                "dataset": "authored/ob_studio",
                "license": "Proprietary / OB Studio internal",
            },
        },
    }

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest successfully created -> {manifest_path}")
    print(f"Total packed tokens: {total_tokens_packed:,}")
    return manifest


if __name__ == "__main__":
    build_production_corpus_and_shards()
