
"""Repeated, alternating-order OpenQuant admission benchmark."""

import json
import time
from pathlib import Path
from statistics import median

import torch

from benchmark_admission_v07 import A, REQUESTS
from openquant.encoder_v07 import OpenQuantEncoder


OUTPUT = Path("reports/admission_balanced_v07.json")


def run_trial(policy):
    encoder = OpenQuantEncoder(
        batch_size=12,
        max_length=128,
        max_graphs=2,
        max_graph_cache_mb=128,
        graph_capture_min_calls=policy,
    )

    # Warm the common eager execution path.
    encoder.encode(A, use_graph=False)
    torch.cuda.synchronize()

    measurements = []

    for name, texts in REQUESTS:
        torch.cuda.synchronize()
        start = time.perf_counter()

        encoder.encode(texts, use_graph=True)

        torch.cuda.synchronize()

        measurements.append({
            "request": name,
            "latency_ms": (
                time.perf_counter() - start
            ) * 1000,
        })

    info = encoder.cache_info()

    result = {
        "policy": policy,
        "request_times": measurements,
        "total_ms": sum(
            x["latency_ms"] for x in measurements
        ),
        "captures": info["stats"].get(
            "graph_captures", 0
        ),
        "evictions": info["stats"].get(
            "graph_evictions", 0
        ),
        "hits": info["stats"].get(
            "graph_hits", 0
        ),
    }

    encoder.clear_graph_cache()

    return result


def main():
    trials = []

    # Alternate which policy executes first.
    orders = [
        (1, 2),
        (2, 1),
        (1, 2),
        (2, 1),
    ]

    for index, order in enumerate(orders, 1):
        print(f"\nTrial group {index}")

        for policy in order:
            result = run_trial(policy)
            trials.append(result)

            print(
                f"  Policy {policy}: "
                f"{result['total_ms']:.3f} ms, "
                f"{result['captures']} captures, "
                f"{result['evictions']} evictions"
            )

    summary = {}

    for policy in (1, 2):
        samples = [
            result for result in trials
            if result["policy"] == policy
        ]

        summary[str(policy)] = {
            "median_total_ms": median(
                x["total_ms"] for x in samples
            ),
            "median_captures": median(
                x["captures"] for x in samples
            ),
            "median_evictions": median(
                x["evictions"] for x in samples
            ),
        }

    old = summary["1"]["median_total_ms"]
    new = summary["2"]["median_total_ms"]

    print("\nBALANCED RESULTS")
    print("Immediate-capture median:", round(old, 3), "ms")
    print("Two-hit median:", round(new, 3), "ms")
    print(
        "Observed reduction:",
        round(100 * (1 - new / old), 2),
        "%",
    )

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "orders": orders,
        "trials": trials,
        "summary": summary,
        "note": (
            "Includes request inference and capture, "
            "excludes model initialization. "
            "Exploratory benchmark; CUDA compilation "
            "and background GPU activity may influence results."
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Report:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
