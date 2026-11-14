import torch
import torch.nn.functional as F
import triton
import triton.testing
from tabulate import tabulate

from flash_attention.reference import attention_ref


def compute_attention_tflops(
    b: int, h_q: int, n_q: int, n_k: int, d: int, causal: bool, latency_ms: float
) -> float:
    """
    Computes effective TFLOPS/s following the
    official FlashAttention-2 paper convention:
    - Forward Matmul 1 (Q @ K^T): 2 * B * H_q * N_q * N_k * D
    - Forward Matmul 2 (P @ V):   2 * B * H_q * N_q * N_k * D
    Total = 4 * B * H_q * N_q * N_k * D (halved if causal).
    """
    flops = 4.0 * b * h_q * n_q * n_k * d
    if causal:
        flops *= 0.5
    return (flops / (latency_ms * 1e-3)) / 1e12


def measure_peak_memory_and_latency(fn, *args, **kwargs):
    """Measures peak allocated memory (MB) and L2-flushed median latency (ms)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        # Single pass to capture exact peak memory allocation
        _ = fn(*args, **kwargs)
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

        # L2-cache-flushed benchmark via Triton's testing harness
        ms = triton.testing.do_bench(
            lambda: fn(*args, **kwargs), quantiles=[0.5, 0.2, 0.8]
        )
        median_ms = ms[0]
        return peak_mem_mb, median_ms
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def run_baseline_suite():
    device_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    print(f"Device: {device_name} (Compute Capability SM{cc_major}{cc_minor})")

    b, h_q, h_kv, d = 4, 16, 16, 64
    dtype = torch.float16
    causal = True
    seq_lens = [512, 1024, 2048, 4096, 8192]

    rows = []
    for n in seq_lens:
        try:
            q = torch.randn((b, h_q, n, d), device="cuda", dtype=dtype)
            k = torch.randn((b, h_kv, n, d), device="cuda", dtype=dtype)
            v = torch.randn((b, h_kv, n, d), device="cuda", dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            rows.append([n, "OOM", "OOM", "OOM", "OOM"])
            continue

        # 1. Eager Materialized Reference (O(N^2) HBM)
        mem_eager, ms_eager = measure_peak_memory_and_latency(
            attention_ref, q, k, v, causal=causal
        )
        tflops_eager = (
            f"{compute_attention_tflops(b, h_q, n, n, d, causal, ms_eager):.2f}"
            if ms_eager is not None
            else "OOM"
        )
        mem_eager_str = f"{mem_eager:.1f} MB" if mem_eager is not None else "OOM"
        ms_eager_str = f"{ms_eager:.3f} ms" if ms_eager is not None else "OOM"

        # 2. PyTorch Native F.scaled_dot_product_attention (cuDNN / Flash Backend)
        mem_sdpa, ms_sdpa = measure_peak_memory_and_latency(
            F.scaled_dot_product_attention, q, k, v, is_causal=causal
        )
        tflops_sdpa = (
            f"{compute_attention_tflops(b, h_q, n, n, d, causal, ms_sdpa):.2f}"
            if ms_sdpa is not None
            else "OOM"
        )
        mem_sdpa_str = f"{mem_sdpa:.1f} MB" if mem_sdpa is not None else "OOM"
        ms_sdpa_str = f"{ms_sdpa:.3f} ms" if ms_sdpa is not None else "OOM"

        rows.append(
            [
                n,
                mem_eager_str,
                ms_eager_str,
                tflops_eager,
                mem_sdpa_str,
                ms_sdpa_str,
                tflops_sdpa,
            ]
        )

        del q, k, v
        torch.cuda.empty_cache()

    headers = [
        "SeqLen (N)",
        "Eager Mem",
        "Eager Latency",
        "Eager TFLOPS/s",
        "PyTorch SDPA Mem",
        "SDPA Latency",
        "SDPA TFLOPS/s",
    ]
    print("\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n")


if __name__ == "__main__":
    run_baseline_suite()
