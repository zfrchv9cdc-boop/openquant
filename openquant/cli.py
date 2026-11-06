from openquant.model_cli import configure_commands, run_inspect, run_analyze
import torch
import triton
import triton.language as tl


@triton.jit
def _int4_matmul(
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
    # Each GPU program calculates one output tile.
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k_offsets = tl.arange(0, BK)

    acc = tl.full((BM, BN), 0, tl.float32)

    # Process input dimensions in small tiles.
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + k_offsets

        # Load input activations (FP16).
        x = tl.load(
            X + m[:, None] * K + k[None, :],
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0,
        )

        # Each byte contains two INT4 weights.
        packed = tl.load(
            Q + n[:, None] * (KP // 2)
              + k[None, :] // 2,
            mask=(n[:, None] < N) & (k[None, :] < K),
            other=0,
        ).to(tl.int32)

        # Extract the correct 4-bit value.
        shift = (k[None, :] & 1) * 4

        nibble = (packed >> shift) & 15

        # Convert unsigned nibble to signed integer.
        signed = tl.where(
            nibble < 8, nibble, nibble - 16
        )

        # Load only the scales needed for this tile.
        scales = tl.load(
            S + n[:, None] * (KP // GROUP)
              + k[None, :] // GROUP,
            mask=(n[:, None] < N) & (k[None, :] < K),
            other=0,
        )

        # Dequantize inside the GPU program.
        weights = (
            signed.to(tl.float32) * scales
        ).to(tl.float16)

        # Matrix multiplication with FP32 accumulation.
        acc = tl.dot(x, tl.trans(weights), acc)

    # Store completed output.
    tl.store(
        Y + m[:, None] * N + n[None, :],
        acc,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


def int4_linear(x, packed, scales, out_features, group_size, *, block_m=64, block_n=64, block_k=64, num_warps=4):
    """Fused groupwise INT4 dequantization + matrix multiplication."""

    if x.ndim != 2 or x.dtype != torch.float16:
        raise ValueError("Input must be a 2D FP16 tensor")

    if not x.is_cuda or not packed.is_cuda or not scales.is_cuda:
        raise ValueError("All tensors must be on CUDA")

    if not x.is_contiguous():
        raise ValueError("Input must be contiguous")

    if packed.dtype != torch.uint8:
        raise ValueError("Packed weights must be uint8")

    if scales.dtype != torch.float32:
        raise ValueError("Scales must be float32")

    if group_size < 2 or group_size % 2:
        raise ValueError("Group size must be even and >= 2")

    m, k = x.shape
    n = out_features

    kp = triton.cdiv(k, group_size) * group_size
    groups = kp // group_size

    if m < 1 or n < 1 or k < 1:
        raise ValueError("Matrix dimensions must be positive")

    if scales.shape != (n, groups):
        raise ValueError("Incorrect scales shape")

    if packed.numel() != n * kp // 2:
        raise ValueError("Incorrect packed weight count")

    if not packed.is_contiguous() or not scales.is_contiguous():
        raise ValueError("Weights and scales must be contiguous")

    if packed.device != x.device or scales.device != x.device:
        raise ValueError("All tensors must use the same GPU")

    output = torch.empty(
        (m, n), device=x.device, dtype=torch.float32
    )

    bm, bn, bk = block_m, block_n, block_k

    grid = (
        triton.cdiv(m, bm),
        triton.cdiv(n, bn),
    )

    _int4_matmul[grid](
        x, packed, scales, output,
        m, n, k, kp, group_size,
        bm, bn, bk,
        num_warps=num_warps,
    )

    return output


