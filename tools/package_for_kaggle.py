"""Packages reviewed Tot-360M sources, tokenizer, and uint16 shards for Kaggle TPU execution."""

import os
import tarfile
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "qa" / "kaggle_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "tot_360m_package.tar.gz"

    files_to_pack = [
        "tot_360m/config.py",
        "tot_360m/src/model.py",
        "tot_360m/src/checkpoint.py",
        "tot_360m/src/train_tpu_calibration.py",
        "tot_360m/tokenizer/production/tokenizer.json",
        "tot_360m/data/manifest.json",
        "tot_360m/tools/codex_runner_safety.py",
    ]

    # Add all 16 shards
    for i in range(16):
        files_to_pack.append(f"tot_360m/data/shards/shard_{i:03d}.bin")

    print(f"Packaging {len(files_to_pack)} verified assets into {tar_path}...")
    with tarfile.open(tar_path, "w:gz", compresslevel=6) as tar:
        for rel_path in files_to_pack:
            abs_p = root / rel_path
            assert abs_p.exists(), f"Missing required file: {abs_p}"
            tar.add(abs_p, arcname=rel_path)

    tar_size = tar_path.stat().st_size
    h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    print(f"Created archive: {tar_size:,} bytes ({tar_size / 1_048_576:.2f} MB)")
    print(f"Archive SHA256:  {h}")


if __name__ == "__main__":
    main()
