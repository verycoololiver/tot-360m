"""Packages reviewed Tot-360M sources, tokenizer, and uint16 shards for Kaggle TPU execution."""

import os
import tarfile
import hashlib
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "qa" / "kaggle_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "tot_360m_package.tar.gz"

    required_rel_files = [
        "__init__.py",
        "config.py",
        "src/__init__.py",
        "src/model.py",
        "src/checkpoint.py",
        "src/train_tpu_calibration.py",
        "src/train_gpu_calibration.py",
        "tools/export_gguf.py",
        "tools/probe_quantizer.py",
        "tokenizer/production/tokenizer.json",
        "tokenizer/production/manifest.json",
    ]

    optional_rel_files = ["data/manifest.json"] + [f"data/shards/shard_{i:03d}.bin" for i in range(16)]
    files_to_pack = list(required_rel_files)
    for rel in optional_rel_files:
        if (repo_root / rel).exists():
            files_to_pack.append(rel)

    print(f"Packaging {len(files_to_pack)} verified assets into {tar_path}...")
    with tarfile.open(tar_path, "w:gz", compresslevel=6) as tar:
        for rel_path in files_to_pack:
            abs_p = repo_root / rel_path
            assert abs_p.exists(), f"Missing required file: {abs_p}"
            tar.add(abs_p, arcname=f"tot_360m/{rel_path}")

    tar_size = tar_path.stat().st_size
    h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    print(f"Created archive: {tar_size:,} bytes ({tar_size / 1_048_576:.2f} MB)")
    print(f"Archive SHA256:  {h}")


if __name__ == "__main__":
    main()
