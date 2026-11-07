
"""Compare immediate and two-hit CUDA Graph admission."""

import json
import time
from pathlib import Path

import torch

from openquant.encoder_v07 import OpenQuantEncoder


REPORT = Path("reports/graph_admission_v07.json")

A = ["A GPU accelerates neural networks."]
B = ["Quantization improves memory efficiency."] * 4
C = ["Models process information using matrix operations."] * 12
D = ["CUDA Graphs reduce repeated launch overhead."] * 2

REQUESTS = [
    ("A_first", A),
    ("B_cold", B),
    ("C_cold", C),
    ("A_second", A),
    ("A_third", A),
    ("D_cold", D),
    ("A_fourth", A),
]


def run_policy(min_calls):
    encoder = OpenQuantEncoder(
        batch_size=12,
        max_length=128,
        max_graphs=2,
        max_graph_cache_mb=128,
        graph_capture_min_calls=min_calls,
    )

    # Warm eager execution before measuring requests.
    encoder.encode(A, use_graph=False)
    torch.cuda.synchronize()

    records = []

    for name, texts in REQUESTS:
        torch.cuda.synchronize()
        start = time.perf_counter()

        actual = encoder.encode(texts, use_graph=True)

        torch.cuda.synchronize()
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        # Correctness checking is excluded from timing.
        expected = encoder.encode(
            texts, use_graph=False
        )

        torch.testing.assert_close(
            actual, expected,
            atol=0.001,
            rtol=0.001,
        )

        info = encoder.cache_info()

        records.append({
            "request": name,
            "latency_ms": elapsed_ms,
            "cached_graphs": info["cached_graphs"],
            "accounted_graph_mib": (
                info["graph_cache_bytes"] / 1024**2
            ),
            "statistics": info["stats"],
        })

        print(
            f"{name:<14}"
            f"{elapsed_ms:>10.3f} ms "
            f"graphs={info['cached_graphs']}"
        )

    result = {
        "capture_min_calls": min_calls,
        "requests": records,
        "final_cache": encoder.cache_info(),
        "total_request_latency_ms": sum(
            item["latency_ms"] for item in records
        ),
    }

    encoder.clear_graph_cache()

    return result


def main():
    print("OPENQUANT GRAPH ADMISSION BENCHMARK")

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "policies": {},
    }

    for minimum in (1, 2):
        print(f"\nPolicy: capture after {minimum} request(s)")

        report["policies"][str(minimum)] = (
            run_policy(minimum)
        )

    old = report["policies"]["1"]
    new = report["policies"]["2"]

    old_stats = old["final_cache"]["stats"]
    new_stats = new["final_cache"]["stats"]

    print("\nSUMMARY")
    print("-" * 60)

    print(
        "Immediate captures:",
        old_stats.get("graph_captures", 0)
    )
    print(
        "Two-hit captures:",
        new_stats.get("graph_captures", 0)
    )

    print(
        "Immediate evictions:",
        old_stats.get("graph_evictions", 0)
    )
    print(
        "Two-hit evictions:",
        new_stats.get("graph_evictions", 0)
    )

    print(
        "Immediate request time:",
        round(old["total_request_latency_ms"], 3),
        "ms",
    )
    print(
        "Two-hit request time:",
        round(new["total_request_latency_ms"], 3),
        "ms",
    )

    REPORT.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("REPORT:", REPORT.resolve())


if __name__ == "__main__":
    main()

