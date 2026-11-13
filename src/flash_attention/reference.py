from __future__ import annotations

import math

import torch

RCP_LN2: float = 1.4426950408889634
LN2: float = 0.6931471805599453


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    """
    Validate tensor ranks, devices, dtypes and GQA head compatibility
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"Expected 4D tensors (B, H, N, D), got shapes: "
            f"Q = {q.shape}, K = {v.shape}, V = {v.shape}"
        )

    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("All input tensors must reside on a CUDA device.")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"Dtype mismatch: Q={q.dtype}, K={k.dtype}, V={v.dtype}")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype: {q.dtype}. Expected fp16, bf16, or fp32.")

    b, h_q, n_q, d = q.shape
    b_k, h_kv, n_k, d_k = k.shape
    b_v, h_v, n_v, d_v = v.shape

    if (
        (b != b_k or b != b_v)
        or (n_k != n_v)
        or (d != d_k or d != d_v)
        or (h_kv != h_v)
    ):
        raise ValueError(f"Incompatible shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    if h_q % h_kv != 0:
        raise ValueError(
            f"Query heads ({h_q}) must be an integer multiple of KV"
            f" heads ({h_kv}) for GQA/MQA."
        )

    return b, h_q, h_kv, n_q, n_k, d


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    upcast_to_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _b, h_q, h_kv, n_q, n_k, d = _validate_inputs(q, k, v)
    orig_dtype = q.dtype
    compute_dtype = torch.float64 if upcast_to_fp64 else orig_dtype

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    q_c = q.to(compute_dtype)
    k_c = k.to(compute_dtype)
    v_c = v.to(compute_dtype)

    if h_q != h_kv:
        num_queries_per_kv = h_q // h_kv
        k_c = k_c.repeat_interleave(num_queries_per_kv, dim=1)
        v_c = v_c.repeat_interleave(num_queries_per_kv, dim=1)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * sm_scale

    if causal:
        row_idx = torch.arange(n_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(n_k, device=q.device).unsqueeze(0)

        causal_mask = col_idx > (row_idx + n_k - n_q)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    lse = torch.logsumexp(
        scores.to(torch.float64 if upcast_to_fp64 else torch.float32), dim=-1
    )

    attn_probs = torch.softmax(
        scores, dim=-1, dtype=torch.float64 if upcast_to_fp64 else torch.float32
    )
    attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

    out = torch.matmul(attn_probs.to(compute_dtype), v_c)
    return out.to(orig_dtype), lse.to(torch.float32)


def flash_attention_v2_tiled_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 128,
    block_n: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, h_q, h_kv, n_q, n_k, d = _validate_inputs(q, k, v)
    orig_dtype = q.dtype

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    qk_scale = sm_scale * RCP_LN2
    num_queries_per_kv = h_q // h_kv
    out = torch.empty_like(q)
    lse = torch.empty((b, h_q, n_q), device=q.device, dtype=torch.float32)

    for i_start in range(0, n_q, block_m):
        i_end = min(i_start + block_m, n_q)
        br = i_end - i_start

        q_tile = q[:, :, i_start:i_end, :]
        m_i = torch.full(
            (b, h_q, br, 1), float("-inf"), device=q.device, dtype=torch.float32
        )
        l_i = torch.zeros((b, h_q, br, 1), device=q.device, dtype=torch.float32)
        acc_o = torch.zeros((b, h_q, br, d), device=q.device, dtype=torch.float32)

        # Causal upper bound for the inner loop:
        # skip all K/V blocks strictly above the diagonal
        max_col_limit = (i_end + n_k - n_q) if causal else n_k

        for j_start in range(0, max_col_limit, block_n):
            j_end = min(j_start + block_n, n_k)
            _bc = j_end - j_start

            k_tile = k[:, :, j_start:j_end, :]  # (B, H_kv, bc, D)
            v_tile = v[:, :, j_start:j_end, :]  # (B, H_kv, bc, D)

            if h_q != h_kv:
                k_tile = k_tile.repeat_interleave(num_queries_per_kv, dim=1)
                v_tile = v_tile.repeat_interleave(num_queries_per_kv, dim=1)

            # 1. Compute scaled tile scores in base-2 log space
            s_ij = torch.matmul(q_tile, k_tile.transpose(-2, -1)).to(torch.float32)
            s_ij *= qk_scale
            # Apply intra-tile causal mask ONLY on
            # boundary blocks intersecting the diagonal
            if causal and (j_end > (i_start + n_k - n_q)):
                row_idx = torch.arange(i_start, i_end, device=q.device).unsqueeze(1)
                col_idx = torch.arange(j_start, j_end, device=q.device).unsqueeze(0)
                mask = col_idx > (row_idx + n_k - n_q)
                s_ij = s_ij.masked_fill(mask, float("-inf"))

            # 2. Online Softmax update in base-2
            m_ij = torch.max(s_ij, dim=-1, keepdim=True).values
            m_next = torch.maximum(m_i, m_ij)

            # Guard against (-inf) - (-inf) = NaN when an entire row is masked
            m_next_safe = torch.where(m_next == float("-inf"), 0.0, m_next)
            m_i_safe = torch.where(m_i == float("-inf"), 0.0, m_i)

            # alpha = 2^(m_i - m_next)
            alpha = torch.exp2(m_i_safe - m_next_safe)
            # If m_i was -inf, alpha should be 0
            alpha = torch.where(m_i == float("-inf"), 0.0, alpha)

            # p_tilde = 2^(s_ij - m_next)
            p_tilde = torch.exp2(s_ij - m_next_safe)

            # 3. Rescale running denominator and output accumulator
            l_i = l_i * alpha + torch.sum(p_tilde, dim=-1, keepdim=True)
            acc_o = acc_o * alpha
            acc_o += torch.matmul(p_tilde.to(orig_dtype), v_tile).to(torch.float32)

            m_i = m_next

        # Epilogue: normalize output and compute natural-base LogSumExp L_i
        # Handle rows that were completely masked (l_i == 0)
        inv_l = torch.where(l_i == 0.0, 0.0, 1.0 / l_i)
        acc_o = acc_o * inv_l

        # Convert base-2 logsumexp back to base-e: L_i = (m_i + log2(l_i)) * ln(2)
        lse_i = (m_i + torch.log2(l_i)) * LN2
        lse_i = torch.where(m_i == float("-inf"), float("-inf"), lse_i)

        out[:, :, i_start:i_end, :] = acc_o.to(orig_dtype)
        lse[:, :, i_start:i_end] = lse_i.squeeze(-1)

    return out, lse
