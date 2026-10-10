"""Verify the native Windows acceleration stack on the current NVIDIA GPU."""

from __future__ import annotations

import importlib.metadata
import os
import time

os.environ.setdefault("FA_VERSION", "2")
os.environ.setdefault("SDNQ_USE_TORCH_COMPILE", "1")
os.environ.setdefault("SDNQ_USE_TRITON_MM", "1")

import torch
import torch.nn.functional as functional

torch.set_float32_matmul_precision("high")

import flash_attn
import sdnq  # noqa: F401 - initializes SDNQ's backend selection
import triton
from flash_attn import flash_attn_func
from sdnq.sdnext import shared


def benchmark(callable_, iterations: int = 20) -> float:
    for _ in range(5):
        callable_()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this virtual environment")

    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0):
        raise RuntimeError(
            "This local FlashAttention wheel is specialized for sm_120; "
            f"the selected GPU reports {capability}."
        )
    if not shared.opts.sdnq_dequantize_compile:
        raise RuntimeError("SDNQ did not enable its Triton/torch.compile path")

    torch.manual_seed(123)
    q = torch.randn((1, 1024, 32, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def flash_call(causal: bool):
        return flash_attn_func(q, k, v, causal=causal)

    def sdpa_call(causal: bool):
        return functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=causal,
        ).transpose(1, 2)

    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(0)} capability={capability}")
    print(
        f"flash-attn={flash_attn.__version__} "
        f"triton-windows={importlib.metadata.version('triton-windows')} "
        f"triton={triton.__version__} sdnq={importlib.metadata.version('sdnq')}"
    )
    print("sdnq_triton_compile=True")

    with torch.inference_mode():
        for causal in (False, True):
            actual = flash_call(causal)
            expected = sdpa_call(causal)
            max_error = (actual.float() - expected.float()).abs().max().item()
            if max_error > 0.01:
                raise RuntimeError(
                    f"FlashAttention numerical check failed: max error {max_error}"
                )
            flash_ms = benchmark(lambda: flash_call(causal))
            sdpa_ms = benchmark(lambda: sdpa_call(causal))
            print(
                f"causal={causal} max_error={max_error:.6f} "
                f"flash_ms={flash_ms:.3f} sdpa_ms={sdpa_ms:.3f} "
                f"speedup={sdpa_ms / flash_ms:.2f}x"
            )


if __name__ == "__main__":
    main()
