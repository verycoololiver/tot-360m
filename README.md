# Tot-360M

[![Project Page](https://img.shields.io/badge/Project_Page-obstudio.org%2Ftot--360m-2563eb)](https://obstudio.org/tot-360m/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](./LICENSE)
[![Status: Pretraining in Progress](https://img.shields.io/badge/Status-Pretraining_in_Progress_(2.84B_%2F_7.2B_Tokens)-amber)](https://obstudio.org/tot-360m/)

> **⚠️ Important Status Notice — Pretraining in Progress:**
> **Tot-360M is currently in bulk pretraining (`2,839,362,984` tokens completed across 7 batches; Batch 8 queued for `~3.40B` tokens).**
> So far, it has trained on **2,839,362,984 tokens** (~39.4% of its ~7.2B-token Chinchilla pretraining target) across Kaggle TPU v5e-8 (`seq_len=1024`) and 2× T4 (`seq_len=4096`) runs, plus initial SFT branch probes (`sft-v1`, `sft-v2`). Early `.gguf` checkpoints published during pretraining are raw, unfinished base snapshots.

**Live status, interactive loss curve, and intermediate `.gguf` snapshots:** **[https://obstudio.org/tot-360m/](https://obstudio.org/tot-360m/)**

---

## Overview

**Tot-360M** is a **345,359,296-parameter** (`314,639,296` non-embedding) multilingual (English / Korean / Spanish) and code language model being trained from random weight initialization on 8-core **Kaggle Cloud TPU v5e-8** and **2× T4 GPU** sessions. It is designed to eventually serve as the local, offline model inside **[Tot](https://obstudio.org/tools/tot/)** on everyday laptops and Raspberry Pi hardware.

Rather than fine-tuning an existing pretrained model, Tot-360M starts from scratch with its own **32,000-token byte-level BPE tokenizer** while adopting a tensor and metadata layout compatible with stock `llama.cpp` (`qwen3` architecture tag) so exported `.gguf` files run natively in `llama.cpp`, Ollama, and LM Studio without custom C++ patches.

### Training Logs & Write-Ups on OB Studio
- **[Tot-360M Technical Specification & Live Pretraining Log](https://obstudio.org/tot-360m/)**
- **[Run 1 Log: 405 Million Tokens on a Kaggle TPU](https://obstudio.org/blog/tot-360m-first-real-training-run/)**
- **[Run 2 Mid-Run Notes](https://obstudio.org/blog/tot-360m-second-training-run/)**
- **[Run 2 Results & First GGUF Checkpoints (917.6M Tokens)](https://obstudio.org/blog/2nd-training-run-done/)**

---

## Architecture Specification

Defined in [`config.py`](./config.py) and implemented in [`src/model.py`](./src/model.py):

| Component | Specification |
| :--- | :--- |
| **Architecture class** | Decoder-only Transformer (`qwen3` GGUF layout) |
| **Total parameters** | `345,359,296` (tied input/output token embeddings) |
| **Non-embedding parameters** | `314,639,296` (32 Transformer decoder blocks) |
| **Layers (`num_hidden_layers`)** | `32` |
| **Hidden size (`hidden_size`)** | `960` |
| **FFN intermediate (`intermediate_size`)** | `2,560` (SwiGLU MLP, $8/3\times$ expansion) |
| **Attention heads** | `15` query heads / `5` KV heads (3:1 GQA, `head_dim = 64`) |
| **QK-Norm** | Shared 64-element RMSNorm vectors per layer (`q_norm`, `k_norm`) |
| **Positional encoding** | NeoX split-half RoPE ($\theta = 10,000.0$, context table `4,096`) |
| **Normalization** | Pre-RMSNorm ($\epsilon = 10^{-6}$) |
| **Vocabulary size** | `32,000` (custom byte-level BPE, [`tokenizer/production/`](./tokenizer/production/)) |
| **Training sequence length** | `1,024` (TPU v5e-8) / `4,096` (2× T4 long-context interleaved batches) |
| **Auxiliary stabilization** | Logit z-loss ($\lambda = 10^{-4}$) + optional NCM conditioning corruption (`--input-corrupt-prob`) |

---

## In-Progress Pretraining Status (2.84B / 7.2B Target Tokens)

Pretraining is underway on Kaggle TPU v5e-8 (`bfloat16` mixed precision with `fp32` master weights and AdamW optimizer states) and 2× T4 (`4,096` context). Intermediate checkpoints are published under Apache-2.0 at **[obstudio.org/tot-360m](https://obstudio.org/tot-360m/)**:

| Run | Optimizer Steps | Cumulative Tokens | Loss (2k Avg / Final) | Throughput | Notes / Snapshots |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Run 1 (Batch 1, TPU v5e-8)** | `439 → 24,785` | `405,680,880` | `4.74 → 2.73` | `~17.7k tok/s` | [Q4_K_M (241 MiB)](https://obstudio.org/downloads/tot_360m_run1_q4_k_m.gguf) · [Q8_0 (351 MiB)](https://obstudio.org/downloads/tot_360m_run1_q8_0.gguf) |
| **Run 2 (Batch 2, TPU v5e-8)** | `24,785 → 56,058` | `917,557,344` | `3.18 → 2.81` | `~17.6k tok/s` | [Q4_K_M (241 MiB)](https://obstudio.org/downloads/tot_360m_run2_q4_k_m.gguf) · [Q8_0 (351 MiB)](https://obstudio.org/downloads/tot_360m_run2_q8_0.gguf) |
| **Run 3 (Batch 3, TPU v5e-8)** | `56,058 → 87,395` | `1,430,481,360` | `2.61 → 2.20` | `~17.5k tok/s` | [Q4_K_M (241 MiB)](https://obstudio.org/downloads/tot_360m_run3_q4_k_m.gguf) · [Q8_0 (351 MiB)](https://obstudio.org/downloads/tot_360m_run3_q8_0.gguf) |
| **Run 3.2 (Batch 3-p2, TPU v5e-8)** | `87,395 → 95,816` | `1,568,316,288` | `final 2.2974` | `~17.1k tok/s` | Completed remaining Batch 3 shards |
| **Run 4 (Batch 4, 2× T4 4K)** | `95,816 → 101,591` | `1,662,910,788` | `final 2.1322` | `~2.3k tok/s` | First 4,096-context expansion batch |
| **Run 5 (Batch 5, 2× T4 4K)** | `101,591 → 109,958` | `1,799,962,248` | `final 1.5644` | `~3.4k tok/s` | 4,096-context + HTML5 games & bilingual ChatML |
| **Run 6 (Batch 6, TPU v5e-8)** | `109,958 → 141,864` | `2,322,199,656` | `2.11 avg / 1.3729 final` | `~17.1k tok/s` | `+522.2M` unseen tokens (`1.82` min 2k-step avg) |
| **Run 7 (Batch 7, TPU v5e-8)** | `141,864 → 173,460` | `2,839,362,984` | `final 1.0576` | `~16.9k tok/s` | `+517.2M` unseen tokens (`30,725s` TPU v5e-8) |
| **Run 8 (Batch 8, TPU v5e-8)** | `173,460 → ~205,000` | `~3,360,000,000` | *Queued on TPU* | — | `~600M` unseen shards + NCM `--input-corrupt-prob 0.03` |

> **Note on running intermediate `.gguf` files:** Because pretraining is not finished yet and SFT has not started, these checkpoints act strictly as raw document/code completers:
> ```bash
> ./llama-cli -m tot_360m_run3_q4_k_m.gguf -p "The gravitational constant" -n 128 --temp 0.7 --repeat-penalty 1.15
> ```

---

## Repository Structure

```text
tot-360m/
├── config.py                        # Frozen Tot360MConfig (345.4M), Tot1024Config, and TotMiniConfig
├── src/
│   ├── model.py                     # PyTorch causal LM (GQA 3:1, shared 64-dim QK-Norm, SwiGLU, NeoX RoPE)
│   ├── checkpoint.py                # Deterministic atomic checkpointing & config/RNG/cursor verification
│   ├── train_tpu_calibration.py     # 8-core PyTorch/XLA TPU v5e-8 pretraining runner (xmp.spawn)
│   └── train_gpu_calibration.py     # Single/multi-GPU CUDA fallback runner
├── tokenizer/
│   ├── build_production_tokenizer.py          # 32,000-vocab byte-level BPE tokenizer builder
│   ├── build_production_corpus_and_shards.py  # Packed uint16 binary shard packer & manifest generator
│   └── production/                            # Production 32k tokenizer.json, vocab.json, and manifest.json
├── tools/
│   ├── export_gguf.py               # Direct PyTorch-to-GGUF exporter (llama.cpp qwen3 layout)
│   ├── package_for_kaggle.py        # Bundles source + tokenizer for Kaggle TPU dataset attachment
│   └── probe_quantizer.py           # 256-element GGUF quantization block alignment analyzer
├── models/                          # Calibration summaries for Run 1, Run 2, and Run 3
└── tests/                           # Unit tests for architecture, BF16/FP16 SDPA, checkpoint recovery & TPU runner
```

---

## Quickstart & Local Verification

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Architecture & Run Unit Tests

You can verify the exact parameter count (`345,359,296`) on PyTorch's `meta` device (0 bytes RAM allocated) and run the full test suite covering `bfloat16`/`float16` attention, deterministic checkpoint recovery, and TPU runner logic:

```bash
pytest
```

### 3. Export a PyTorch Checkpoint to GGUF

```bash
python tools/export_gguf.py \
  --config baseline \
  --checkpoint path/to/checkpoint.pt \
  --tokenizer tokenizer/production/tokenizer.json \
  --out tot_360m_f16.gguf \
  --dtype f16
```

---

## License

Code, tokenizer, and intermediate model weights are released under the **[Apache License 2.0](./LICENSE)**.
Built by **Oliver** at **[OB Studio (obstudio.org)](https://obstudio.org/tot-360m/)**.
