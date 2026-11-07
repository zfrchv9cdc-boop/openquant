
"""Measure OpenQuant v0.7 graph-cache admission and eviction."""

import argparse
import json
from pathlib import Path

import torch

from openquant.encoder_v07 import OpenQuantEncoder


OUTPUT = Path("reports/graph_budget_v07.json")
MIB = 1024 ** 2


def memory_snapshot():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()

    return {
        "allocated_mib": torch.cuda.memory_allocated() / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
        "free_mib": free / MIB,
        "total_mib": total / MIB,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--budget-mb", type=float, default=80
    )

    parser.add_argument(
        "--max-graphs", type=int, default=2
    )

    args = parser.parse_args()

    if args.budget_mb < 0 or args.max_graphs < 0:
        parser.error("Invalid cache settings")

    encoder = OpenQuantEncoder(
        batch_size=12,
        max_length=128,
        max_graphs=args.max_graphs,
        max_graph_cache_mb=args.budget_mb,
        min_capture_free_mb=256,
    )

    cases = {
        "short_single": [
            "A GPU accelerates machine learning."
        ],
        "medium_batch4": [
            "Quantization reduces model memory requirements."
        ] * 4,
        "long_batch12": [
            "Neural network inference involves matrix "
            "multiplication and activation functions."
        ] * 12,
        "short_single_again": [
            "A GPU accelerates machine learning."
        ],
    }

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "budget_mib": args.budget_mb,
        "max_graphs": args.max_graphs,
        "before": memory_snapshot(),
        "cases": {},
    }

    for name, sentences in cases.items():
        expected = encoder.encode(
            sentences, use_graph=False
        )

        actual = encoder.encode(
            sentences, use_graph=True
        )

        torch.testing.assert_close(
            actual, expected, rtol=0.001, atol=0.001
        )

        info = encoder.cache_info()

        assert info["cached_graphs"] <= args.max_graphs
        assert (
            info["graph_cache_bytes"]
            <= info["graph_budget_bytes"]
        )

        report["cases"][name] = {
            "correctness": "PASS",
            "cache": info,
            "memory": memory_snapshot(),
        }

        print("\n", name)
        print("  Correctness: PASS")
        print("  Cached graphs:", info["cached_graphs"])
        print(
            "  Accounted graph memory:",
            round(info["graph_cache_bytes"] / MIB, 3),
            "MiB",
        )
        print("  Statistics:", info["stats"])

    encoder.clear_graph_cache()

    report["after_clear"] = memory_snapshot()
    report["cache_after_clear"] = encoder.cache_info()

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nBudgeted graph memory test: PASS")
    print("Report:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
