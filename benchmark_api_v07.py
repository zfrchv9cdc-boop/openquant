
"""OpenQuant v0.7 end-to-end text encoding benchmark."""

import json
import time
from pathlib import Path
from statistics import median

import torch

from openquant.encoder_v07 import OpenQuantEncoder


REPORT = Path("reports/encoder_api_v07.json")

SENTENCES = [
    "The weather is beautiful today.",
    "It is a sunny and pleasant day.",
    "A computer executes mathematical instructions.",
    "The GPU accelerates neural network computations.",
    "Quantization compresses neural network weights.",
    "Low precision can reduce memory requirements.",
    "The astronaut traveled to the Moon.",
    "A spacecraft was launched into orbit.",
    "The hospital building is completely empty.",
    "The doctor is examining a patient.",
    "Jordan is a country in the Middle East.",
    "Amman is the capital city of Jordan.",
]


def benchmark(fn, repeats=30):
    samples = []

    for _ in range(5):
        fn()

    torch.cuda.synchronize()

    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()

        fn()

        torch.cuda.synchronize()
        samples.append(
            (time.perf_counter() - start) * 1000
        )

    ordered = sorted(samples)

    return {
        "median_ms": median(samples),
        "p90_ms": ordered[
            max(0, int(0.9 * len(ordered)) - 1)
        ],
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    print("OPENQUANT v0.7 - END-TO-END ENCODER")
    print("GPU:", torch.cuda.get_device_name(0))

    load_started = time.perf_counter()

    encoder = OpenQuantEncoder(
        batch_size=12,
        max_length=128,
        max_graphs=1,
        graph_max_tokens=2048,
    )

    load_seconds = (
        time.perf_counter() - load_started
    )

    scenarios = {
        "single_short": SENTENCES[:1],
        "batch4_short": SENTENCES[:4],
        "batch12_short": SENTENCES,
        "batch12_long": [
            " ".join([sentence] * 12)
            for sentence in SENTENCES
        ],
    }

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "model_load_seconds": load_seconds,
        "scope": (
            "Tokenizer, host-to-device transfer, "
            "model forward, pooling, graph input copy, "
            "and device-to-host embedding output. "
            "Model loading excluded."
        ),
        "results": {},
    }

    for name, sentences in scenarios.items():
        print("\n" + "-" * 65)
        print("WORKLOAD:", name)
        print("Sentences:", len(sentences))

        # Release earlier captured graphs.
        encoder.clear_graph_cache()

        eager = benchmark(
            lambda: encoder.encode(
                sentences, use_graph=False
            )
        )

        eager_result = encoder.encode(
            sentences, use_graph=False
        )

        allocated_before_graph = (
            torch.cuda.memory_allocated()
        )

        reserved_before_graph = (
            torch.cuda.memory_reserved()
        )

        # First invocation captures the shape.
        graph_result = encoder.encode(
            sentences, use_graph=True
        )

        torch.testing.assert_close(
            eager_result,
            graph_result,
            rtol=0.001,
            atol=0.001,
        )

        allocated_after_graph = (
            torch.cuda.memory_allocated()
        )

        reserved_after_graph = (
            torch.cuda.memory_reserved()
        )

        graph = benchmark(
            lambda: encoder.encode(
                sentences, use_graph=True
            )
        )

        speedup = (
            eager["median_ms"]
            / graph["median_ms"]
        )

        reduction = (
            1 - graph["median_ms"] / eager["median_ms"]
        ) * 100

        info = encoder.cache_info()

        record = {
            "batch_size": len(sentences),
            "eager": eager,
            "graph": graph,
            "speedup": speedup,
            "latency_reduction_pct": reduction,
            "correctness": "PASS",
            "cached_graphs": info["cached_graphs"],
            "graph_allocated_delta_bytes": (
                allocated_after_graph
                - allocated_before_graph
            ),
            "graph_reserved_delta_bytes": (
                reserved_after_graph
                - reserved_before_graph
            ),
        }

        report["results"][name] = record

        print(
            f"Eager: {eager['median_ms']:.3f} ms"
        )
        print(
            f"Graph: {graph['median_ms']:.3f} ms"
        )
        print(f"Speedup: {speedup:.2f}x")
        print(f"Latency reduction: {reduction:.2f}%")
        print("Correctness: PASS")

        if info["cached_graphs"] == 0:
            print(
                "NOTE: Graph unavailable; "
                "the graph-enabled request used eager fallback."
            )

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
