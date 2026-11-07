"""Measure CUDA allocator behavior across graph lifecycles."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from benchmark_capture_peak_v07 import CASES
from openquant.encoder_v07 import OpenQuantEncoder


MIB = 1024 ** 2


def snap():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()

    return {
        "allocated_mib": round(
            torch.cuda.memory_allocated() / MIB, 3
        ),
        "reserved_mib": round(
            torch.cuda.memory_reserved() / MIB, 3
        ),
        "free_device_mib": round(free / MIB, 3),
        "total_device_mib": round(total / MIB, 3),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=("eager", "immediate", "two-hit"),
        required=True,
    )
    parser.add_argument(
        "--cycles", type=int, default=8
    )
    parser.add_argument(
        "--budget-mb", type=float, default=128
    )

    args = parser.parse_args()

    if args.cycles < 2 or args.budget_mb < 0:
        parser.error(
            "Use at least two cycles and a nonnegative budget"
        )

    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")

    min_calls = (
        2 if args.mode == "two-hit" else 1
    )

    use_graph = args.mode != "eager"

    encoder = OpenQuantEncoder(
        batch_size=12,
        max_length=128,
        max_graphs=1,
        max_graph_cache_mb=args.budget_mb,
        graph_capture_min_calls=min_calls,
        min_capture_free_mb=256,
    )

    references = [
        encoder.encode(texts, use_graph=False)
        for _, texts in CASES
    ]

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    baseline = snap()
    rows = []

    for cycle in range(1, args.cycles + 1):
        encoder.clear_graph_cache()

        before = snap()

        stats_before = dict(
            encoder.cache_info()["stats"]
        )

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        for (_, texts), reference in zip(
            CASES, references
        ):
            result = encoder.encode(
                texts,
                use_graph=use_graph,
            )

            torch.testing.assert_close(
                result,
                reference,
                rtol=0.001,
                atol=0.001,
            )

            del result

        torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        peaks = {
            "allocated_mib": round(
                torch.cuda.max_memory_allocated() / MIB,
                3,
            ),
            "reserved_mib": round(
                torch.cuda.max_memory_reserved() / MIB,
                3,
            ),
        }

        info = encoder.cache_info()

        assert info["cached_graphs"] <= 1

        assert (
            info["graph_cache_bytes"]
            <= info["graph_budget_bytes"]
        )

        encoder.clear_graph_cache()
        gc.collect()
        torch.cuda.empty_cache()

        after = snap()

        stats_after = encoder.cache_info()["stats"]

        events = {
            key: (
                stats_after.get(key, 0)
                - stats_before.get(key, 0)
            )
            for key in (
                "graph_captures",
                "graph_evictions",
                "graph_hits",
                "budget_rejections",
                "blocked_shape_fallbacks",
                "cold_shape_fallbacks",
            )
        }

        row = {
            "cycle": cycle,
            "elapsed_ms": round(elapsed_ms, 3),
            "before": before,
            "peak": peaks,
            "after_clear": after,
            "events": events,
        }

        rows.append(row)

        print(
            f"Cycle {cycle}: {elapsed_ms:.1f} ms; "
            f"peak_alloc={peaks['allocated_mib']:.1f} MiB; "
            f"after_clear_alloc={after['allocated_mib']:.1f} MiB; "
            f"after_clear_reserved={after['reserved_mib']:.1f} MiB"
        )

    clean = [
        row["after_clear"]["allocated_mib"]
        for row in rows
    ]

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "mode": args.mode,
        "cycles": args.cycles,
        "baseline": baseline,
        "observations": rows,
        "summary": {
            "median_after_clear_allocated_mib": (
                statistics.median(clean)
            ),
            "after_clear_first_to_last_delta_mib": (
                round(clean[-1] - clean[0], 3)
            ),
            "max_peak_allocated_mib": max(
                row["peak"]["allocated_mib"]
                for row in rows
            ),
        },
        "note": (
            "Memory trends are diagnostics, not "
            "proof of leaks. PyTorch allocator metrics "
            "do not capture all external GPU allocations."
        ),
    }

    output = Path(
        f"reports/memory_stability_v07_{args.mode}.json"
    )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Correctness: PASS")
    print(
        "First-to-last after-clear allocated delta:",
        report["summary"][
            "after_clear_first_to_last_delta_mib"
        ],
        "MiB",
    )
    print("Report:", output.resolve())


if __name__ == "__main__":
    main()