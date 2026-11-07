
"""Experimental fused INT8 dequantization and GEMM."""

import json
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
import triton.testing

from openquant.triton_int8 import (
    int8_linear as production_int8
)

from openquant.quantization import (
    quantize,
    dequantize,
)


@triton.jit
def int8_gemm_kernel(
    X, Q, S, Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)

    scales = tl.load(
        S + cols,
        mask=cols < N,
        other=0,
    )

    accumulator = tl.full(
        (BM, BN), 0, tl.float32
    )

    for index in range(tl.cdiv(K, BK)):
        k = index * BK + ks

        x = tl.load(
            X + rows[:, None] * K + k[None, :],
            mask=(rows[:, None] < M) &
                 (k[None, :] < K),
            other=0,
        )

        q = tl.load(
            Q + cols[:, None] * K + k[None, :],
            mask=(cols[:, None] < N) &
                 (k[None, :] < K),
            other=0,
        )

        weights = (
            q.to(tl.float32) * scales[:, None]
        ).to(tl.float16)

        accumulator = tl.dot(
            x,
            tl.trans(weights),
            accumulator,
        )

    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        accumulator,
        mask=(rows[:, None] < M) &
             (cols[None, :] < N),
    )


def fused_int8(x, packed, scales, n):
    m, k = x.shape

    output = torch.empty(
        (m, n),
        device=x.device,
        dtype=torch.float32,
    )

    grid = (
        triton.cdiv(m, 32),
        triton.cdiv(n, 32),
    )

    int8_gemm_kernel[grid](
        x, packed, scales, output,
        m, n, k,
        32, 32, 64,
        num_warps=4,
    )

    return output


def measure(fn):
    return float(
        triton.testing.do_bench(
            fn,
            warmup=10,
            rep=30,
            return_mode="median",
        )
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    rng = np.random.default_rng(42)
    results = []

    print("GPU:", torch.cuda.get_device_name(0))
    print("Production INT8 versus existing decode-per-call")
    print()

    for n in (384, 1536):
        k = 384

        weights = rng.normal(
            size=(n, k)
        ).astype(np.float32)

        compressed = quantize(weights, bits=8)

        q = torch.from_numpy(
            compressed.data
        ).cuda()

        scales = torch.from_numpy(
            compressed.scales
        ).cuda().contiguous().reshape(-1)

        reconstructed = torch.from_numpy(
            dequantize(compressed)
        ).cuda()

        for m in (1, 4, 16, 128):
            x = torch.randn(
                (m, k),
                device="cuda",
                dtype=torch.float16,
            )

            reference = (
                x.float() @ reconstructed.T
            )

            def fused():
                return production_int8(x, q, scales)

            def decode_each_time():
                full = (
                    q.float() *
                    scales[:, None]
                )
                return x.float() @ full.T

            actual = fused()
            torch.cuda.synchronize()

            torch.testing.assert_close(
                actual,
                reference,
                rtol=0.02,
                atol=0.05,
            )

            fused_ms = measure(fused)
            decoded_ms = measure(decode_each_time)

            speedup = decoded_ms / fused_ms

            row = {
                "M": m,
                "N": n,
                "K": k,
                "fused_ms": fused_ms,
                "decode_each_ms": decoded_ms,
                "speedup": speedup,
                "correctness": "PASS",
            }

            results.append(row)

            print(
                f"M={m:3d} N={n:4d} "
                f"fused={fused_ms:.6f} ms "
                f"decode={decoded_ms:.6f} ms "
                f"speedup={speedup:.2f}x "
                "PASS"
            )

    output = Path(
        "reports/fused_int8_v07.json"
    )
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "results": results,
                "timing": "Non-CUDA-Graph",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Correctness: {len(results)} / 8 passed")
    print("Report:", output.resolve())


if __name__ == "__main__":
    main()
