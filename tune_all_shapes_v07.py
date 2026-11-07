
"""Correctness-first benchmark of OpenQuant INT4 kernels."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import triton.testing

from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)
from openquant.triton_int4 import int4_linear


def measure(fn, warmup=5, rep=20):
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help="Test both output widths and group sizes",
    )

    parser.add_argument(
        "--rep",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if args.rep < 1:
        parser.error("Repetition time must be positive")

    if not torch.cuda.is_available():
        parser.error("CUDA required")

    rng = np.random.default_rng(42)

    batches = (1, 4, 16, 128)
    widths = (384, 1536) if args.full else (1536,)
    groups = (16, 32) if args.full else (32,)
    k = 384

    rows = []
    failures = []
    began = time.perf_counter()

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Timing: non-CUDA-Graph")

    for group in groups:
        for n in widths:
            weights = rng.normal(
                size=(n, k)
            ).astype(np.float32)

            quantized = quantize_int4_groupwise(
                weights, group
            )

            packed = torch.from_numpy(
                quantized.data
            ).cuda()

            scales = torch.from_numpy(
                quantized.scales
            ).cuda()

            restored = torch.from_numpy(
                dequantize_int4_groupwise(quantized)
            ).cuda().half()

            original = torch.from_numpy(
                weights
            ).cuda().half()

            for m in batches:
                x = torch.randn(
                    (m, k),
                    dtype=torch.float16,
                    device="cuda",
                )

                reference = (
                    x @ restored.T
                ).float()

                fp16_ms = measure(
                    lambda: x @ original.T,
                    rep=args.rep,
                )

                algorithms = [
                    "legacy",
                    "packed",
                ]

                if m <= 4:
                    algorithms.extend([
                        "gemv",
                        "batched_bn4",
                        "batched_bn8",
                        "batched_bn16",
                    ])

                timings = {}

                print(
                    f"\nM={m} N={n} K={k} G={group}"
                )
                print(
                    f"FP16 dense: {fp16_ms:.6f} ms"
                )

                for algorithm in algorithms:
                    def call(alg=algorithm):
                        if alg.startswith("batched_bn"):
                            block = int(alg.rsplit("bn", 1)[1])
                            return int4_linear(
                                x, packed, scales, n, group,
                                algorithm="batched",
                                block_n=block,
                            )

                        return int4_linear(
                            x, packed, scales, n, group,
                            algorithm=alg,
                        )

                    try:
                        result = call()

                        torch.cuda.synchronize()

                        torch.testing.assert_close(
                            result,
                            reference,
                            rtol=0.02,
                            atol=0.02,
                        )

                        latency = measure(
                            call, rep=args.rep
                        )

                        timings[algorithm] = latency

                        print(
                            f"  {algorithm:8} "
                            f"{latency:.6f} ms PASS"
                        )

                    except Exception as exc:
                        failures.append({
                            "M": m,
                            "N": n,
                            "K": k,
                            "group": group,
                            "algorithm": algorithm,
                            "error": repr(exc),
                        })

                        print(
                            f"  {algorithm:8} "
                            f"FAIL: {str(exc)[:150]}"
                        )

                winner = (
                    min(timings, key=timings.get)
                    if timings else None
                )

                best_ms = (
                    timings[winner]
                    if winner else None
                )

                row = {
                    "M": m,
                    "N": n,
                    "K": k,
                    "group": group,
                    "fp16_ms": fp16_ms,
                    "int4_timings_ms": timings,
                    "best_int4": winner,
                    "best_int4_ms": best_ms,
                    "fp16_over_best_int4": (
                        fp16_ms / best_ms
                        if best_ms else None
                    ),
                }

                rows.append(row)

                print(
                    "Fastest correct INT4:",
                    winner,
                )

    elapsed = time.perf_counter() - began

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "full_matrix": args.full,
        "benchmark_wall_seconds": elapsed,
        "timing_method": (
            "triton.testing.do_bench without CUDA Graph"
        ),
        "rows": rows,
        "failures": failures,
    }

    path = Path(
        "reports/kernel_sweep_v07.json"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nKERNEL SWEEP FINISHED")
    print("Cases:", len(rows))
    print("Failed configurations:", len(failures))
    print("Report:", path.resolve())

    if failures:
        raise SystemExit(
            "One or more kernel configurations failed. "
            "Inspect the report before tuning."
        )


if __name__ == "__main__":
    main()
