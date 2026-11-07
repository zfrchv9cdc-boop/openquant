"""Compare optional capture-budget floor policies."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from benchmark_capture_peak_v07 import CASES
from openquant.encoder_v07 import OpenQuantEncoder


EVENTS = (
    "graph_captures",
    "graph_evictions",
    "graph_hits",
    "budget_rejections",
    "preflight_budget_fallbacks",
    "capture_failures",
    "blocked_shape_fallbacks",
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--budget-mb", type=float, default=16
    )
    parser.add_argument(
        "--capture-floor-mb",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--cycles", type=int, default=4
    )

    args = parser.parse_args()

    if (
        not math.isfinite(args.budget_mb)
        or not math.isfinite(args.capture_floor_mb)
        or min(args.budget_mb, args.capture_floor_mb) < 0
        or args.cycles < 1
    ):
        parser.error("Invalid budget, floor or cycle count")

    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")

    encoder = OpenQuantEncoder(
        batch_size=12,
        max_length=128,
        max_graphs=1,
        max_graph_cache_mb=args.budget_mb,
        graph_capture_min_budget_mb=args.capture_floor_mb,
        min_capture_free_mb=256,
        graph_capture_min_calls=1,
    )

    expected = [
        encoder.encode(texts, use_graph=False)
        for _, texts in CASES
    ]

    torch.cuda.synchronize()
    trials = []

    for cycle in range(1, args.cycles + 1):
        encoder.clear_graph_cache()

        counters_before = encoder.cache_info()["stats"]

        torch.cuda.synchronize()
        started = time.perf_counter()
        samples = []

        for (name, texts), reference in zip(
            CASES, expected
        ):
            torch.cuda.synchronize()
            began = time.perf_counter()

            actual = encoder.encode(
                texts, use_graph=True
            )

            torch.cuda.synchronize()
            elapsed = (
                time.perf_counter() - began
            ) * 1000

            torch.testing.assert_close(
                actual,
                reference,
                rtol=0.001,
                atol=0.001,
            )

            samples.append({
                "request": name,
                "latency_ms": round(elapsed, 3),
            })

        total_ms = (
            time.perf_counter() - started
        ) * 1000

        counters_after = encoder.cache_info()["stats"]

        delta = {
            key: (
                counters_after.get(key, 0)
                - counters_before.get(key, 0)
            )
            for key in EVENTS
        }

        trials.append({
            "cycle": cycle,
            "total_ms": round(total_ms, 3),
            "requests": samples,
            "events": delta,
        })

        print(
            f"Cycle {cycle}: {total_ms:.3f} ms; "
            f"{delta}"
        )

    median_total = statistics.median(
        row["total_ms"] for row in trials
    )

    summary = {
        "median_total_ms": median_total,
        "event_totals": {
            key: sum(
                row["events"][key]
                for row in trials
            )
            for key in EVENTS
        },
    }

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "budget_mib": args.budget_mb,
        "capture_floor_mib": args.capture_floor_mb,
        "cycles": args.cycles,
        "trials": trials,
        "summary": summary,
        "note": (
            "Includes cold capture, fallback and replay. "
            "Excludes encoder initialization. "
            "Separate runs may have different GPU clocks "
            "and compilation conditions."
        ),
    }

    output = Path(
        "reports/"
        f"capture_floor_budget{args.budget_mb:g}"
        f"_floor{args.capture_floor_mb:g}.json"
    )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Median total latency: {median_total:.3f} ms"
    )
    print("Correctness: PASS")
    print("Report:", output.resolve())


if __name__ == "__main__":
    main()