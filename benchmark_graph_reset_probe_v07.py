"""Compare default and explicit CUDA Graph destruction."""

import argparse
import gc
import json
import weakref
from pathlib import Path

import torch

from benchmark_capture_peak_v07 import CASES
from openquant.encoder_v07 import OpenQuantEncoder

MIB = 1024 ** 2


class ProbeEncoder(OpenQuantEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_weakrefs = []

    def _evict_lru(self):
        cached = next(iter(self.graphs.values()))
        self.graph_weakrefs.append(weakref.ref(cached))
        return super()._evict_lru()

    def clear_graph_cache(self):
        self.graph_weakrefs.extend(
            weakref.ref(graph)
            for graph in self.graphs.values()
        )
        return super().clear_graph_cache()


class ExplicitResetEncoder(ProbeEncoder):
    def _evict_lru(self):
        torch.cuda.synchronize()

        cached = next(iter(self.graphs.values()))
        cached.graph.reset()

        return super()._evict_lru()

    def clear_graph_cache(self):
        with self.lock:
            torch.cuda.synchronize()

            for cached in self.graphs.values():
                cached.graph.reset()

            return super().clear_graph_cache()


def memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "allocated_mib": round(
            torch.cuda.memory_allocated() / MIB, 3
        ),
        "reserved_mib": round(
            torch.cuda.memory_reserved() / MIB, 3
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["original", "explicit"],
        required=True,
    )

    parser.add_argument(
        "--cycles", type=int, default=8
    )

    args = parser.parse_args()

    if args.cycles < 2:
        parser.error("At least two cycles required")

    cls = (
        ExplicitResetEncoder
        if args.mode == "explicit"
        else ProbeEncoder
    )

    encoder = cls(
        batch_size=12,
        max_length=128,
        max_graphs=1,
        max_graph_cache_mb=128,
        graph_capture_min_calls=1,
    )

    references = [
        encoder.encode(texts, use_graph=False)
        for _, texts in CASES
    ]

    baseline = memory()
    records = []

    for cycle in range(1, args.cycles + 1):
        encoder.clear_graph_cache()

        for (_, texts), reference in zip(
            CASES, references
        ):
            actual = encoder.encode(
                texts,
                use_graph=True,
            )

            torch.testing.assert_close(
                actual, reference,
                rtol=0.001,
                atol=0.001,
            )

            del actual

        encoder.clear_graph_cache()

        current = memory()

        alive = sum(
            ref() is not None
            for ref in encoder.graph_weakrefs
        )

        row = {
            "cycle": cycle,
            **current,
            "live_python_graph_wrappers": alive,
        }

        records.append(row)

        print(
            f"Cycle {cycle}: "
            f"allocated={current['allocated_mib']:.3f} MiB, "
            f"reserved={current['reserved_mib']:.3f} MiB, "
            f"live_wrappers={alive}"
        )

    drift = (
        records[-1]["allocated_mib"]
        - records[0]["allocated_mib"]
    )

    report = {
        "mode": args.mode,
        "gpu": torch.cuda.get_device_name(0),
        "baseline": baseline,
        "cycles": records,
        "first_to_last_allocated_drift_mib": round(
            drift, 3
        ),
        "note": (
            "Live wrapper counts indicate Python references. "
            "They do not independently identify CUDA allocator "
            "or graph-private pool ownership."
        ),
    }

    output = Path(
        f"reports/graph_reset_probe_{args.mode}.json"
    )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Correctness: PASS")
    print("First-to-last drift:", round(drift, 3), "MiB")
    print("Report:", output.resolve())


if __name__ == "__main__":
    main()