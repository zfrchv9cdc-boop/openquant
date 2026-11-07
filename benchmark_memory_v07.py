
"""OpenQuant v0.7: CUDA Graph memory audit."""

import json
from pathlib import Path

import torch

from openquant.encoder_v07 import OpenQuantEncoder


REPORT = Path("reports/graph_memory_v07.json")
MIB = 1024 ** 2


def snapshot():
    torch.cuda.synchronize()

    free, total = torch.cuda.mem_get_info()

    return {
        "allocated_mib": (
            torch.cuda.memory_allocated() / MIB
        ),
        "reserved_mib": (
            torch.cuda.memory_reserved() / MIB
        ),
        "device_free_mib": free / MIB,
        "device_total_mib": total / MIB,
    }


def main():
    encoder = OpenQuantEncoder(
        batch_size=12,
        max_length=128,
        max_graphs=2,
        graph_max_tokens=2048,
    )

    scenarios = {
        "single_short": [
            "The computer processes information."
        ],
        "batch4_medium": [
            "An artificial intelligence model processes text."
        ] * 4,
        "batch12_long": [
            "Machine learning uses mathematical operations "
            "to extract patterns from information."
        ] * 12,
    }

    baseline = snapshot()

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "baseline": baseline,
        "scenarios": {},
    }

    for name, sentences in scenarios.items():
        print("\nSCENARIO:", name)

        encoder.encode(
            sentences, use_graph=False
        )

        before = snapshot()

        encoder.encode(
            sentences, use_graph=True
        )

        after = snapshot()

        info = encoder.cache_info()

        record = {
            "before": before,
            "after": after,
            "cached_graphs": info["cached_graphs"],
            "cache_statistics": info["stats"],
            "allocated_delta_mib": (
                after["allocated_mib"]
                - before["allocated_mib"]
            ),
            "reserved_delta_mib": (
                after["reserved_mib"]
                - before["reserved_mib"]
            ),
        }

        report["scenarios"][name] = record

        print("Cached graphs:", info["cached_graphs"])
        print(
            "Allocated delta:",
            round(record["allocated_delta_mib"], 3),
            "MiB",
        )
        print(
            "Reserved delta:",
            round(record["reserved_delta_mib"], 3),
            "MiB",
        )

        assert info["cached_graphs"] <= 2

    encoder.clear_graph_cache()

    report["after_cache_clear"] = snapshot()
    report["final_cache"] = encoder.cache_info()

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nREPORT:", REPORT.resolve())


if __name__ == "__main__":
    main()
