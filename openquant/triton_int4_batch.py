"""Experimental weight-sharing INT4 GEMV for small input batches."""
import torch
import triton
import triton.language as tl


@triton.jit
def _batched_gemv_kernel(
    X, Q, S, Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    GROUP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
):
    m = tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    b = tl.arange(0, BC)

    k0 = 2 * b
    k1 = k0 + 1

    # Load packed weights once for all input rows.
    p = tl.load(
        Q + n[:, None] * (KP // 2) + b[None, :],
        mask=(n[:, None] < N) & (k0[None, :] < K),
        other=0,
    ).to(tl.int32)

    low = p & 15
    high = (p >> 4) & 15

    low = tl.where(low < 8, low, low - 16)
    high = tl.where(high < 8, high, high - 16)

    # Same scale for both weights in each byte.
    scales = tl.load(
        S + n[:, None] * (KP // GROUP)
          + k0[None, :] // GROUP,
        mask=(n[:, None] < N) & (k0[None, :] < K),
        other=0,
    )

    w0 = (
        low.to(tl.float32) * scales
    ).to(tl.float16).to(tl.float32)

    w1 = (
        high.to(tl.float32) * scales
    ).to(tl.float16).to(tl.float32)

    # Load every input row in the same GPU program.
    x0 = tl.load(
        X + m[:, None] * K + k0[None, :],
        mask=(m[:, None] < M) & (k0[None, :] < K),
        other=0,
    ).to(tl.float32)

    x1 = tl.load(
        X + m[:, None] * K + k1[None, :],
        mask=(m[:, None] < M) & (k1[None, :] < K),
        other=0,
    ).to(tl.float32)

    # Broadcast inputs against weights:
    # [M, 1, K/2] * [1, N, K/2].
    partial = (
        x0[:, None, :] * w0[None, :, :]
        + x1[:, None, :] * w1[None, :, :]
    )

    result = tl.sum(partial, axis=2)

    tl.store(
        Y + m[:, None] * N + n[None, :],
        result,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


def batched_int4_linear(
    x, packed, scales, out_features, group_size,
    *, block_n=8, num_warps=4
):
    m, k = x.shape

    if not 1 <= m <= 4:
        raise ValueError("Batched GEMV supports M=1..4")

    if k > 2048:
        raise ValueError("Batched GEMV currently supports K<=2048")

    if block_n not in (2, 4, 8, 16, 32):
        raise ValueError("Unsupported output block")

    kp = triton.cdiv(k, group_size) * group_size
    bc = triton.next_power_of_2(triton.cdiv(k, 2))

    output = torch.empty(
        (m, out_features),
        device=x.device,
        dtype=torch.float32,
    )

    _batched_gemv_kernel[
        (triton.cdiv(out_features, block_n),)
    ](
        x, packed, scales, output,
        m, out_features, k, kp, group_size,
        triton.next_power_of_2(m),
        block_n,
        bc,
        num_warps=num_warps,
    )

    return output
