import pytest
import torch

from flash_attention.reference import attention_ref, flash_attention_v2_tiled_ref


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("h_q,h_kv", [(8, 8), (8, 2), (8, 1)])  # MHA, GQA, MQA
@pytest.mark.parametrize("n_q,n_k", [(128, 128), (257, 257), (512, 512), (128, 256)])
@pytest.mark.parametrize("d", [64, 128])
def test_tiled_reference_accuracy(
    dtype: torch.dtype,
    causal: bool,
    h_q: int,
    h_kv: int,
    n_q: int,
    n_k: int,
    d: int,
) -> None:
    torch.manual_seed(0)
    device = "cuda"
    b = 2

    q = torch.randn((b, h_q, n_q, d), device=device, dtype=dtype)
    k = torch.randn((b, h_kv, n_k, d), device=device, dtype=dtype)
    v = torch.randn((b, h_kv, n_k, d), device=device, dtype=dtype)

    # 1. Exact FP64 Ground Truth
    out_ref, lse_ref = attention_ref(q, k, v, causal=causal, upcast_to_fp64=True)

    # 2. Baseline PyTorch low-precision (FP16/BF16)
    out_pt, lse_pt = attention_ref(q, k, v, causal=causal, upcast_to_fp64=False)

    # 3. FlashAttention-2 Tiled Implementation
    out_fa, lse_fa = flash_attention_v2_tiled_ref(
        q, k, v, causal=causal, block_m=64, block_n=64
    )

    # Compute max errors against FP64 ground truth
    pt_max_err = (out_pt - out_ref).abs().max().item()
    fa_max_err = (out_fa - out_ref).abs().max().item()

    # Dao-style numerical assertion:
    # FlashAttention error must be <= 2x baseline PyTorch quantization error
    assert fa_max_err <= 2.0 * pt_max_err + 1e-5, (
        f"FlashAttention error ({fa_max_err:.6e}) exceeded 2x PyTorch baseline "
        f"error ({pt_max_err:.6e})"
    )

    # Verify LogSumExp state tensor (used in backward pass)
    valid_lse = torch.isfinite(lse_ref)
    pt_lse_err = (lse_pt[valid_lse] - lse_ref[valid_lse]).abs().max().item()
    fa_lse_err = (lse_fa[valid_lse] - lse_ref[valid_lse]).abs().max().item()

    assert fa_lse_err <= 2.0 * pt_lse_err + 1e-4, (
        f"LogSumExp error ({fa_lse_err:.6e}) exceeded 2x PyTorch baseline "
        f"error ({pt_lse_err:.6e})"
    )
