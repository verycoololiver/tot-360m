"""GGML Quantizer Probe and Analytical Tensor Payload Auditor for Tot Models.

Pinned Reference:
- llama.cpp immutable commit: 3d65c90d04d337e88f2b1f7f0061f40a5324e662 (Release b11515)
- Quantizer source: src/llama-quant.cpp (llama_tensor_quantize_internal)
- Block specs: ggml/include/ggml.h / ggml-quants.h

Audits exact GGML ne[0] block divisibility and analytical tensor payload sizes under:
- Pure Q4_K target with stock Q5_0 fallback (when ne[0] % 256 != 0 and ne[0] % 32 == 0)
- Stock F16 fallback if ne[0] % 32 != 0
- FP32 normalization vectors (1D tensors)
- Single tied embedding storage (deduplicating lm_head and embed_tokens via PyTorch parameter identity)

DISCLAIMER: This script calculates analytical tensor payloads under the stated uniform policy.
It does NOT execute a native llama-quantize binary or measure CPU inference throughput.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch

from tot_360m.config import Tot360MConfig, Tot1024Config
from tot_360m.src.model import TotForCausalLM


@dataclass
class DynamicTensorSpec:
    pytorch_name: str
    shape_pytorch: Tuple[int, ...]
    numel: int
    is_norm: bool

    @property
    def ne0(self) -> int:
        """Innermost/contiguous dimension in GGML (input features for 2D, length for 1D)."""
        if self.is_norm:
            return self.numel
        return self.shape_pytorch[1]

    @property
    def ne1(self) -> int:
        """Second dimension in GGML (output features for 2D, 1 for 1D)."""
        if self.is_norm:
            return 1
        return self.shape_pytorch[0]


def extract_tensor_specs_from_model(config) -> List[DynamicTensorSpec]:
    """Dynamically extracts all unique parameters from TotForCausalLM instantiated on meta device."""
    with torch.device("meta"):
        model = TotForCausalLM(config)

    specs = []
    seen_params = set()

    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id in seen_params:
            # Skip tied tensor (e.g. lm_head.weight sharing storage with model.embed_tokens.weight)
            continue
        seen_params.add(param_id)

        is_norm = (param.dim() == 1)
        specs.append(DynamicTensorSpec(
            pytorch_name=name,
            shape_pytorch=tuple(param.shape),
            numel=param.numel(),
            is_norm=is_norm,
        ))

    return specs


def audit_quantization(candidate_name: str, config):
    print(f"\n{'='*75}")
    print(f"ANALYTICAL QUANTIZATION AUDIT: {candidate_name}")
    print(f"Reference: llama.cpp commit 09b3fa47d488f28fa8e803c46aa1d4d0398fb914")
    print(f"{'='*75}")

    tensors = extract_tensor_specs_from_model(config)
    
    total_params = sum(t.numel for t in tensors)
    norm_params = sum(t.numel for t in tensors if t.is_norm)
    matrix_params = total_params - norm_params
    
    total_bytes = 0
    type_counts: Dict[str, int] = {}
    type_bytes: Dict[str, int] = {}
    type_params: Dict[str, int] = {}
    fallback_tensors = []
    
    for t in tensors:
        if t.is_norm:
            q_type = "F32"
            b = t.numel * 4  # 4 bytes per FP32
        else:
            # Check GGML ne[0] divisibility
            if t.ne0 % 256 == 0:
                q_type = "Q4_K"
                b = (t.numel // 256) * 144  # 144 bytes per 256 weights
            elif t.ne0 % 32 == 0:
                q_type = "Q5_0"  # llama-quant fallback for ne0 % 256 != 0
                b = (t.numel // 32) * 22   # 22 bytes per 32 weights
                fallback_tensors.append((t.pytorch_name, t.ne0, t.shape_pytorch))
            else:
                q_type = "F16"  # llama-quant fallback for non-32 divisible
                b = t.numel * 2
                fallback_tensors.append((t.pytorch_name, t.ne0, t.shape_pytorch))
                
        total_bytes += b
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
        type_bytes[q_type] = type_bytes.get(q_type, 0) + b
        type_params[q_type] = type_params.get(q_type, 0) + t.numel

    print(f"Total Unique Model Parameters: {total_params:,}")
    print(f"  - Matrix Weights:            {matrix_params:,}")
    print(f"  - Normalization Scales:      {norm_params:,}")
    print(f"\nCalculated Storage Breakdown by Quantized Type:")
    print(f"{'-'*75}")
    print(f"{'Type':<10} {'Tensors':<10} {'Weights':<15} {'Payload (Bytes)':<18} {'Payload (MB)':<12} {'BPW':<8}")
    print(f"{'-'*75}")
    for q_type, count in sorted(type_counts.items()):
        b = type_bytes[q_type]
        p = type_params[q_type]
        bpw = (b * 8) / p
        print(f"{q_type:<10} {count:<10} {p:<15,} {b:<18,} {b / (1000**2):<12.2f} {bpw:<8.2f}")
    print(f"{'-'*75}")
    print(f"TOTAL TENSOR PAYLOAD:          {total_bytes:,} bytes ({total_bytes / (1000**2):.2f} MB / {total_bytes / (1024**2):.2f} MiB)")
    print(f"EFFECTIVE AVERAGE BPW:         {(total_bytes * 8) / total_params:.2f} bits/weight")
    
    if fallback_tensors:
        print(f"\n[WARNING] FALLBACK DETECTED: {len(fallback_tensors)} tensors fell back to Q5_0 due to ne[0] % 256 != 0:")
        sample_fallbacks = fallback_tensors[:5]
        for name, ne0, shape in sample_fallbacks:
            print(f"   * {name:<40} PyTorch shape: {str(shape):<15} ne[0] = {ne0} ({ne0} % 256 = {ne0 % 256})")
        if len(fallback_tensors) > 5:
            print(f"   ... and {len(fallback_tensors) - 5} more identical fallback tensors.")
    else:
        print(f"\n[OK] 100% ALIGNED: All 2D matrix rows (ne[0]) are divisible by 256. Zero Q5_0 fallbacks.")
        
    return total_params, total_bytes


def main():
    p960, b960 = audit_quantization("Tot-960 (Baseline d_model=960)", Tot360MConfig())
    p1024, b1024 = audit_quantization("Tot-1024 (Aligned d_model=1024)", Tot1024Config())
    
    diff_params = p1024 - p960
    diff_bytes = b1024 - b960
    diff_mb = diff_bytes / (1000**2)
    
    print(f"\n{'='*75}")
    print("ANALYTICAL COMPARATIVE SUMMARY")
    print(f"{'='*75}")
    print(f"Parameter Difference:  +{diff_params:,} (+{diff_params / p960 * 100:.1f}%)")
    print(f"Calculated Payload:    {diff_mb:+.2f} MB ({diff_bytes:+,} bytes)")
    print(f"\nNotes on Analytical Scope:")
    print(f"1. Figures represent calculated tensor payloads under uniform Q4_K with Q5_0 fallback.")
    print(f"2. Excludes GGUF container headers, string metadata, tensor alignment padding,")
    print(f"   and mixed-precision Q4_K_M overrides.")
    print(f"3. Throughput and memory-bandwidth implications remain hypotheses pending real runtime")
    print(f"   benchmarks on the physical Intel i7-6500U CPU.")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
