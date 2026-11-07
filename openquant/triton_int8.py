"""Fused INT8-weight linear operation for CUDA.

Input:   X float16 [M, K]
Weights: Q int8    [N, K]
Scales:  S float32 [N] or [N, 1]
Output:  Y float32 [M, N]

Scales follow openquant.quantization.quantize(bits=8).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_linear_kernel(
    X, Q, S, Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)

    scales = tl.load(
        S + n,
        mask=n < N,
        other=0,
    )

    acc = tl.full(
        (BM, BN),
        0,
        tl.float32,
    )

    for block in range(tl.cdiv(K, BK)):
        k = block * BK + offsets

        activations = tl.load(
            X + m[:, None] * K + k[None, :],
            mask=(m[:, None] < M)
            & (k[None, :] < K),
            other=0,
        )

        integers = tl.load(
            Q + n[:, None] * K + k[None, :],
            mask=(n[:, None] < N)
            & (k[None, :] < K),
            other=0,
        )

        weights = (
            integers.to(tl.float32)
            * scales[:, None]
        ).to(tl.float16)

        acc = tl.dot(
            activations,
            tl.trans(weights),
            acc,
        )

    tl.store(
        Y + m[:, None] * N + n[None, :],
        acc,
        mask=(m[:, None] < M)
        & (n[None, :] < N),
    )


def int8_linear(x, packed, scales):
    """Execute a fused INT8-weight linear layer.

    No dequantized weight matrix is materialized.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if not isinstance(packed, torch.Tensor):
        raise TypeError("packed must be a tensor")
    if not isinstance(scales, torch.Tensor):
        raise TypeError("scales must be a tensor")

    if x.ndim != 2 or packed.ndim != 2:
        raise ValueError(
            "Inputs and weights must be 2D"
        )

    if x.dtype != torch.float16:
        raise ValueError(
            "Activations must be float16"
        )

    if packed.dtype != torch.int8:
        raise ValueError(
            "Weights must be signed int8"
        )

    if scales.dtype != torch.float32:
        raise ValueError(
            "Scales must be float32"
        )

    if not (
        x.is_cuda
        and packed.is_cuda
        and scales.is_cuda
    ):
        raise ValueError(
            "All tensors must be on CUDA"
        )

    if not (
        x.device == packed.device == scales.device
    ):
        raise ValueError(
            "All tensors must use the same GPU"
        )

    if not (
        x.is_contiguous()
        and packed.is_contiguous()
        and scales.is_contiguous()
    ):
        raise ValueError(
            "Tensors must be contiguous"
        )

    m, k = x.shape
    n, weight_k = packed.shape

    if min(m, n, k) < 1:
        raise ValueError(
            "Matrix dimensions must be positive"
        )

    if weight_k != k:
        raise ValueError(
            "Input and weight dimensions differ"
        )

    if scales.shape not in (
        (n,),
        (n, 1),
    ):
        raise ValueError(
            "Expected one scale per output row"
        )

    flat_scales = scales.reshape(n)

    output = torch.empty(
        (m, n),
        dtype=torch.float32,
        device=x.device,
    )

    bm = 32
    bn = 32
    bk = 64

    grid = (
        triton.cdiv(m, bm),
        triton.cdiv(n, bn),
    )

    _int8_linear_kernel[grid](
        x,
        packed,
        flat_scales,
        output,
        m, n, k,
        bm, bn, bk,
        num_warps=4,
    )

    return output