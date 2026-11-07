import torch
import triton
import triton.language as tl


@triton.jit
def _int4_gemv(
    X, Q, S, Y,
    N: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    GROUP: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    byte = tl.arange(0, BC)

    k0 = 2 * byte
    k1 = k0 + 1

    p = tl.load(
        Q + n[:, None] * (KP // 2) + byte[None, :],
        mask=(n[:, None] < N) & (k0[None, :] < K),
        other=0,
    ).to(tl.int32)

    low = p & 15
    high = (p >> 4) & 15

    low = tl.where(low < 8, low, low - 16)
    high = tl.where(high < 8, high, high - 16)

    scale = tl.load(
        S + n[:, None] * (KP // GROUP)
          + k0[None, :] // GROUP,
        mask=(n[:, None] < N) & (k0[None, :] < K),
        other=0,
    )

    x0 = tl.load(
        X + m * K + k0,
        mask=k0 < K,
        other=0,
    ).to(tl.float32)

    x1 = tl.load(
        X + m * K + k1,
        mask=k1 < K,
        other=0,
    ).to(tl.float32)

    w0 = (low.to(tl.float32) * scale).to(tl.float16)
    w1 = (high.to(tl.float32) * scale).to(tl.float16)

    partial = (
        x0[None, :] * w0.to(tl.float32)
        + x1[None, :] * w1.to(tl.float32)
    )

    values = tl.sum(partial, axis=1)

    tl.store(Y + m * N + n, values, mask=n < N)


@triton.jit
def _int4_packed_gemm(
    X, Q, S, Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    GROUP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)

    offsets_k = tl.arange(0, BK)
    offsets_byte = tl.arange(0, BK // 2)

    acc = tl.full((BM, BN), 0, tl.float32)

    for block in range(tl.cdiv(K, BK)):
        k = block * BK + offsets_k
        pair_k = block * BK + 2 * offsets_byte

        x = tl.load(
            X + m[:, None] * K + k[None, :],
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0,
        )

        p = tl.load(
            Q + n[:, None] * (KP // 2)
              + pair_k[None, :] // 2,
            mask=(n[:, None] < N) & (pair_k[None, :] < K),
            other=0,
        ).to(tl.int32)

        low = p & 15
        high = (p >> 4) & 15

        low = tl.where(low < 8, low, low - 16)
        high = tl.where(high < 8, high, high - 16)

        integers = tl.interleave(low, high)

        scale = tl.load(
            S + n[:, None] * (KP // GROUP)
              + pair_k[None, :] // GROUP,
            mask=(n[:, None] < N) & (pair_k[None, :] < K),
            other=0,
        )

        repeated_scale = tl.interleave(scale, scale)

        weights = (
            integers.to(tl.float32) * repeated_scale
        ).to(tl.float16)

        acc = tl.dot(x, tl.trans(weights), acc)

    tl.store(
        Y + m[:, None] * N + n[None, :],
        acc,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


def fast_int4_linear(
    x,
    packed,
    scales,
    out_features,
    group_size,
    *,
    algorithm="gemv",
    block_m=64,
    block_n=64,
    block_k=64,
    num_warps=4,
):
    m, k = x.shape
    n = out_features
    kp = triton.cdiv(k, group_size) * group_size

    result = torch.empty(
        (m, n), device=x.device, dtype=torch.float32
    )

    if algorithm == "gemv":
        if k > 8192:
            raise ValueError("GEMV path currently supports K <= 8192")

        bytes_per_row = triton.next_power_of_2(
            triton.cdiv(k, 2)
        )

        _int4_gemv[
            (m, triton.cdiv(n, 4))
        ](
            x, packed, scales, result,
            n, k, kp, group_size,
            4, bytes_per_row,
            num_warps=4,
        )

    elif algorithm == "packed":
        if block_k < 32 or block_k % 2:
            raise ValueError("Packed GEMM needs block_k >= 32 and even")

        _int4_packed_gemm[
            (
                triton.cdiv(m, block_m),
                triton.cdiv(n, block_n),
            )
        ](
            x, packed, scales, result,
            m, n, k, kp, group_size,
            block_m, block_n, block_k,
            num_warps=num_warps,
        )

    else:
        raise ValueError("Unknown kernel algorithm")

    return result
