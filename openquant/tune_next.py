"""Shape-specific INT4 kernel benchmarking."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import torch
import triton.testing

from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)
from openquant.model_eval import load_layers
from openquant.triton_int4 import int4_linear


PACKED_CONFIGS = [
    (16, 64, 64, 4),
    (32, 64, 64, 4),
    (64, 64, 64, 4),
    (64, 128, 64, 4),
    (64, 64, 128, 4),
    (128, 64, 64, 4),
    (128, 128, 64, 8),
]


def benchmark(fn):
    fn()
    torch.cuda.synchronize()

    results = []

    for _ in range(3):
        ms = triton.testing.do_bench_cudagraph(
            fn, rep=30, return_mode="median"
        )
        results.append(float(ms))

    return median(results)


def candidates(m, x, packed, scales, n, group):
    result = []

    def add(name, **kwargs):
        def run():
            return int4_linear(
                x, packed, scales, n, group, **kwargs
            )
        result.append((name, run, kwargs))

    add("legacy", algorithm="legacy")

    add("packed_default", algorithm="packed")

    if m <= 4:
        add("gemv_original", algorithm="gemv")

        for bn in (2, 4, 8, 16):
            add(
                f"batched_BN{bn}",
                algorithm="batched",
                block_n=bn,
                num_warps=4,
            )

    for bm, bn, bk, warps in PACKED_CONFIGS:
        add(
            f"packed_{bm}_{bn}_{bk}_{warps}",
            algorithm="packed",
            block_m=bm,
            block_n=bn,
            block_k=bk,
            num_warps=warps,
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/minilm/model.safetensors",
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument(
        "--output",
        default="reports/kernel_tuning_v061.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    _, layers = load_layers(args.model)

    if not 0 <= args.layer_index < len(layers):
        raise ValueError("Layer index out of range")

    layer_name, matrix = layers[args.layer_index]
    weights = np.asarray(matrix, dtype=np.float32)
    n, k = weights.shape

    quantized = quantize_int4_groupwise(weights, args.group)

    packed = torch.from_numpy(quantized.data).cuda()
    scales = torch.from_numpy(quantized.scales).cuda()

    restored = torch.from_numpy(
        dequantize_int4_groupwise(quantized)
    ).cuda().half()

    report = {
        "project": "OpenQuant",
        "version": "0.6.1-experimental",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "layer": layer_name,
        "weight_shape": [n, k],
        "group_size": args.group,
        "results": [],
    }

    print("\nOPENQUANT INT4 KERNEL AUTOTUNER")
    print("GPU:", report["gpu"])
    print("Layer:", layer_name)
    print("Shape:", (n, k))

    for m in (1, 4, 16, 128):
        x = torch.randn(
            (m, k),
            device="cuda",
            dtype=torch.float16,
        )

        reference = x @ restored.T

        passing = []
        failed = []

        print("\n" + "=" * 65)
        print("BATCH SIZE:", m)
        print("=" * 65)

        for name, fn, config in candidates(
            m, x, packed, scales, n, args.group
        ):
            try:
                output = fn()
                torch.cuda.synchronize()

                if not torch.isfinite(output).all().item():
                    raise ValueError("Non-finite output")

                torch.testing.assert_close(
                    output.float(),
                    reference.float(),
                    rtol=0.02,
                    atol=0.02,
                )

                elapsed = benchmark(fn)

                passing.append({
                    "name": name,
                    "latency_ms": elapsed,
                    "config": config,
                })

                print(
                    f"{name:<28} {elapsed * 1000:>9.3f} us PASS"
                )

            except Exception as exc:
                reason = str(exc)[:200]
                failed.append({
                    "name": name,
                    "reason": reason,
                })
                print(f"{name:<28} FAILED: {reason}")

        fp16_ms = benchmark(lambda: x @ restored.T)
        passing.sort(key=lambda item: item["latency_ms"])

        baseline = next(
            (
                item for item in passing
                if item["name"] == "legacy"
            ),
            None,
        )

        previous = [
            item for item in passing
            if item["name"] in (
                "legacy",
                "packed_default",
                "gemv_original",
            )
        ]

        best = passing[0] if passing else None
        old_best = min(
            previous,
            key=lambda item: item["latency_ms"],
        ) if previous else None

        if best:
            print("\nFASTEST:", best["name"])
            print(
                f"INT4 latency: {best['latency_ms'] * 1000:.3f} us"
            )

            if baseline:
                print(
                    "Speedup vs legacy: "
                    f"{baseline['latency_ms'] / best['latency_ms']:.2f}x"
                )

            if old_best:
                improvement = (
                    old_best["latency_ms"] / best["latency_ms"]
                )
                print(
                    f"Improvement vs previous best: {improvement:.2f}x"
                )
        else:
            print("No passing candidate for this shape")

        report["results"].append({
            "batch_size": m,
            "fp16_latency_ms": fp16_ms,
            "baseline": baseline,
            "previous_best": old_best,
            "best": best,
            "passing": passing,
            "failed": failed,
        })

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nResults saved to:", destination.resolve())


if __name__ == "__main__":
    main()
