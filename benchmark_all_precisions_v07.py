
"""OpenQuant: common-method FP32/FP16/INT8/INT4 comparison."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import triton.testing

from openquant.quantization import quantize, dequantize
from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)
from openquant.triton_int8 import int8_linear
from openquant.triton_int4 import int4_linear
from openquant.gpu_benchmark import decode_gpu


def timed(fn, rep):
    return float(
        triton.testing.do_bench(
            fn,
            warmup=10,
            rep=rep,
            return_mode="median",
        )
    )


def relative_error(actual, reference):
    return float(
        (
            torch.linalg.vector_norm(
                actual.float() - reference.float()
            )
            / torch.linalg.vector_norm(
                reference.float()
            ).clamp_min(1e-12)
            * 100
        ).item()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--rows", type=int, default=1536)
    parser.add_argument("--cols", type=int, default=384)
    args = parser.parse_args()

    if min(args.rep, args.rows, args.cols) < 1:
        parser.error("All arguments must be positive")

    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.set_float32_matmul_precision("highest")

    rng = np.random.default_rng(42)
    n, k = args.rows, args.cols

    weights_np = rng.standard_normal(
        (n, k)
    ).astype(np.float32)

    weights_fp32 = torch.from_numpy(
        weights_np
    ).cuda()

    weights_fp16 = weights_fp32.half()

    int8 = quantize(weights_np, bits=8)
    int4 = quantize_int4_groupwise(
        weights_np, group_size=32
    )

    q8 = torch.from_numpy(int8.data).cuda()
    s8 = torch.from_numpy(int8.scales).cuda()

    q4 = torch.from_numpy(int4.data).cuda()
    s4 = torch.from_numpy(int4.scales).cuda()

    # References are created once, outside timed regions.
    w8_reference = torch.from_numpy(
        dequantize(int8)
    ).cuda()

    w4_reference = torch.from_numpy(
        dequantize_int4_groupwise(int4)
    ).cuda()

    storage = {
        "FP32": weights_np.nbytes,
        "FP16": weights_np.size * 2,
        "INT8 decode": int8.data.nbytes + int8.scales.nbytes,
        "INT8 fused": int8.data.nbytes + int8.scales.nbytes,
        "INT4 fused": int4.data.nbytes + int4.scales.nbytes,
    }

    results = []
    begin = time.perf_counter()

    print("GPU:", torch.cuda.get_device_name(0))
    print("Matrix:", (n, k))
    print("Benchmark: non-CUDA-Graph GPU timing")
    print("INT4 format: groupwise, group size 32")

    for m in (1, 4, 16, 128):
        inputs_np = rng.standard_normal(
            (m, k)
        ).astype(np.float32)

        x32 = torch.from_numpy(inputs_np).cuda()
        x16 = x32.half()

        fp32_reference = x32 @ weights_fp32.T
        int8_reference = x32 @ w8_reference.T
        int4_reference = x32 @ w4_reference.T

        algorithm = (
            "gemv" if m <= 4 else "packed"
        )

        operations = {
            "FP32": lambda: x32 @ weights_fp32.T,
            "FP16": lambda: x16 @ weights_fp16.T,
            "INT8 decode": lambda: (
                x32 @ decode_gpu(
                    q8, s8, 8, (n, k)
                ).T
            ),
            "INT8 fused": lambda: int8_linear(
                x16, q8, s8
            ),
            "INT4 fused": lambda: int4_linear(
                x16,
                q4,
                s4,
                n,
                32,
                algorithm=algorithm,
            ),
        }

        references = {
            "FP32": fp32_reference,
            "FP16": fp32_reference,
            "INT8 decode": int8_reference,
            "INT8 fused": int8_reference,
            "INT4 fused": int4_reference,
        }

        print("\nBATCH:", m)
        print("-" * 72)

        batch_timings = {}

        for name, operation in operations.items():
            with torch.inference_mode():
                actual = operation()
                torch.cuda.synchronize()

                if not torch.isfinite(actual).all().item():
                    raise AssertionError(
                        f"{name}: nonfinite output"
                    )

                quant_ref = references[name]

                if name in ("INT8 fused", "INT4 fused"):
                    torch.testing.assert_close(
                        actual.float(),
                        quant_ref.float(),
                        rtol=0.02,
                        atol=0.05,
                    )

                error_quant = relative_error(
                    actual, quant_ref
                )
                error_fp32 = relative_error(
                    actual, fp32_reference
                )

                if name == "FP32":
                    assert error_fp32 == 0.0

                elif name == "FP16":
                    assert error_fp32 < 0.5

                if name.startswith("INT8"):
                    assert error_fp32 < 5.0

                if name == "INT4 fused":
                    assert error_fp32 < 25.0

                ms = timed(operation, args.rep)

            batch_timings[name] = ms

            results.append({
                "batch": m,
                "format": name,
                "latency_ms": ms,
                "error_vs_quantized_ref_pct": error_quant,
                "error_vs_fp32_pct": error_fp32,
                "storage_bytes": storage[name],
                "correctness": "PASS",
            })

            print(
                f"{name:<14} "
                f"{ms:>10.5f} ms  "
                f"FP32 error={error_fp32:>7.3f}%  "
                "PASS"
            )

        print(
            "Fused INT8 / decode speedup:",
            round(
                batch_timings["INT8 decode"]
                / batch_timings["INT8 fused"], 2
            ),
            "x",
        )

    assert len(results) == 20

    output = Path(
        "reports/all_precisions_v07.json"
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "shape": [n, k],
        "batches": [1, 4, 16, 128],
        "benchmark_rep_ms": args.rep,
        "method": "triton.testing.do_bench",
        "wall_seconds": time.perf_counter() - begin,
        "results": results,
    }

    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n20 measured configurations: PASS")
    print("Report:", output.resolve())


if __name__ == "__main__":
    main()
