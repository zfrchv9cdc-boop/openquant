"""OpenQuant v0.7 CUDA Graph peak-memory profiler."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from openquant.encoder_v07 import (
    BUCKETS,
    OpenQuantEncoder,
)

MIB = 1024 ** 2

CASES = [
    (
        "short_capture",
        ["CUDA Graphs accelerate inference."],
    ),
    (
        "medium_replacement",
        [" ".join(
            ["quantization", "reduces", "memory"] * 9
        )],
    ),
    (
        "long_replacement",
        [" ".join(
            ["transformers", "execute",
             "attention", "efficiently"] * 20
        )],
    ),
    (
        "short_again",
        ["CUDA Graphs accelerate inference."],
    ),
]


def snapshot():
    torch.cuda.synchronize()

    free, total = torch.cuda.mem_get_info()

    return {
        "allocated_bytes": int(
            torch.cuda.memory_allocated()
        ),
        "reserved_bytes": int(
            torch.cuda.memory_reserved()
        ),
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
    }


def shape_for(encoder, texts):
    tokens = encoder.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=encoder.max_length,
        return_tensors="pt",
    )

    original = tokens["input_ids"].shape[1]

    bucket = next(
        (value for value in BUCKETS
         if value >= original),
        original,
    )

    if bucket > encoder.max_length:
        bucket = original

    return [len(texts), int(bucket)]


def mib(value):
    return round(value / MIB, 3)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--budget-mb",
        type=float,
        default=128,
    )

    parser.add_argument(
        "--max-graphs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--headroom-mb",
        type=float,
        default=256,
    )

    args = parser.parse_args()

    if (
        args.budget_mb < 0
        or args.max_graphs < 0
        or args.headroom_mb < 0
    ):
        parser.error("Memory settings must be nonnegative")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    encoder = OpenQuantEncoder(
        batch_size=12,
        max_length=128,
        max_graphs=args.max_graphs,
        max_graph_cache_mb=args.budget_mb,
        min_capture_free_mb=args.headroom_mb,
        graph_capture_min_calls=1,
    )

    # Warm basic execution before profiling.
    encoder.encode(
        ["OpenQuant warmup sentence."],
        use_graph=False,
    )

    torch.cuda.synchronize()

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "budget_mib": args.budget_mb,
        "max_graphs": args.max_graphs,
        "baseline": snapshot(),
        "cases": [],
    }

    previous_stats = {}

    for name, texts in CASES:
        shape = shape_for(encoder, texts)

        # Reference inference is deliberately excluded
        # from the measured graph-enabled request.
        expected = encoder.encode(
            texts,
            use_graph=False,
        )

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        before = snapshot()

        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        actual = encoder.encode(
            texts,
            use_graph=True,
        )

        torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        peak_allocated = int(
            torch.cuda.max_memory_allocated()
        )

        peak_reserved = int(
            torch.cuda.max_memory_reserved()
        )

        after = snapshot()

        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.001,
            atol=0.001,
        )

        info = encoder.cache_info()

        assert (
            info["graph_cache_bytes"]
            <= info["graph_budget_bytes"]
        )

        assert info["cached_graphs"] <= args.max_graphs

        assert peak_allocated >= before["allocated_bytes"]

        assert peak_reserved >= before["reserved_bytes"]

        stats = info["stats"]

        keys = [
            "graph_captures",
            "graph_evictions",
            "graph_hits",
            "budget_rejections",
            "capture_failures",
            "low_memory_fallbacks",
            "blocked_shape_fallbacks",
        ]

        changes = {
            key: (
                stats.get(key, 0)
                - previous_stats.get(key, 0)
            )
            for key in keys
        }

        previous_stats = dict(stats)

        row = {
            "name": name,
            "shape": shape,
            "latency_ms": round(elapsed_ms, 3),
            "correctness": "PASS",
            "before": before,
            "after": after,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_above_baseline_mib": mib(
                max(
                    0,
                    peak_allocated
                    - before["allocated_bytes"],
                )
            ),
            "peak_reserved_above_baseline_mib": mib(
                max(
                    0,
                    peak_reserved
                    - before["reserved_bytes"],
                )
            ),
            "device_free_change_mib": mib(
                after["device_free_bytes"]
                - before["device_free_bytes"]
            ),
            "accounted_graph_cache_mib": mib(
                info["graph_cache_bytes"]
            ),
            "stats_delta": changes,
            "cache": info,
        }

        report["cases"].append(row)

        print()
        print(name)
        print("  Shape:", shape)
        print("  Correctness: PASS")
        print(
            "  Request latency:",
            round(elapsed_ms, 3),
            "ms",
        )
        print(
            "  Allocator peak:",
            mib(peak_allocated),
            "MiB",
        )
        print(
            "  Reserved peak:",
            mib(peak_reserved),
            "MiB",
        )
        print(
            "  Extra allocated peak:",
            row["peak_allocated_above_baseline_mib"],
            "MiB",
        )
        print(
            "  Accounted graph cache:",
            row["accounted_graph_cache_mib"],
            "MiB",
        )
        print("  Events:", changes)

    shapes = [
        tuple(row["shape"])
        for row in report["cases"][:3]
    ]

    if len(set(shapes)) < 2:
        raise AssertionError(
            "Profiling requires different input shapes"
        )

    encoder.clear_graph_cache()
    gc.collect()

    report["after_clear"] = snapshot()

    budget_label = (
        format(args.budget_mb, "g")
        .replace(".", "p")
    )

    output = Path(
        f"reports/capture_peak_v07_{budget_label}mb.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Peak-memory profiling: PASS")
    print("Report:", output.resolve())
    print()
    print(
        "Note: allocator peaks are process-level "
        "PyTorch measurements, not physical "
        "device-wide peak VRAM."
    )


if __name__ == "__main__":
    main()