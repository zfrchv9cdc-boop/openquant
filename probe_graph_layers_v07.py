
"""Isolate OpenQuant CUDA Graph memory ownership."""

import argparse
import gc
import json
from pathlib import Path

import torch

from openquant.encoder_v07 import (
    FixedShapeGraph,
    OpenQuantEncoder,
)
from openquant.inference_v07 import encode_sentences
from probe_graph_lifecycle_v07 import (
    TinyModel,
    TinyTokenizer,
)

MIB = 1024 ** 2
LENGTHS = (16, 32, 64, 16)


def cleanup():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def memory():
    torch.cuda.synchronize()
    return {
        "allocated_mib": round(
            torch.cuda.memory_allocated() / MIB, 3
        ),
        "reserved_mib": round(
            torch.cuda.memory_reserved() / MIB, 3
        ),
    }


def make_tokens(length):
    return {
        "input_ids": torch.full(
            (1, length),
            7,
            dtype=torch.long,
            device="cuda",
        ),
        "attention_mask": torch.ones(
            (1, length),
            dtype=torch.long,
            device="cuda",
        ),
    }


def direct_capture(model, length, early_release=False):
    tokens = make_tokens(length)

    with torch.inference_mode():
        reference = encode_sentences(model, tokens)

        # Warm the same forward path before capture.
        for _ in range(4):
            encode_sentences(model, tokens)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with torch.inference_mode():
        with torch.cuda.graph(graph):
            captured = encode_sentences(model, tokens)

        graph.replay()
        actual = captured.clone()

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        reference,
        rtol=0.001,
        atol=0.001,
    )

    if early_release:
        # Release captured output before resetting graph.
        del captured
        torch.cuda.synchronize()
        graph.reset()
    else:
        graph.reset()
        del captured

    del actual, reference, graph, tokens


def fixed_capture(model, length, early_release=False):
    tokens = make_tokens(length)

    with torch.inference_mode():
        reference = encode_sentences(model, tokens)

    cached = FixedShapeGraph(model, tokens)

    actual = cached(tokens)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        reference,
        rtol=0.001,
        atol=0.001,
    )

    if early_release:
        # Test a different lifetime ordering.
        cached.output = None
        cached.inputs.clear()
        torch.cuda.synchronize()

    cached.graph.reset()

    del actual, reference, cached, tokens


def run_encoder(encoder):
    samples = (
        ["cat"],
        ["x" * 20],
        ["z" * 40],
        ["cat"],
    )

    for texts in samples:
        expected = encoder.encode(
            texts,
            use_graph=False,
        )

        actual = encoder.encode(
            texts,
            use_graph=True,
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.001,
            atol=0.001,
        )

        del expected, actual

    encoder.clear_graph_cache()


def allocator_state():
    snapshot = torch.cuda.memory_snapshot()

    states = {}
    sizes = []

    for segment in snapshot:
        for block in segment.get("blocks", []):
            size = int(block.get("size", 0))
            state = block.get("state", "unknown")

            states[state] = states.get(state, 0) + size

            if state == "active_allocated":
                sizes.append(size)

    return {
        "states_mib": {
            key: round(value / MIB, 3)
            for key, value in states.items()
        },
        "largest_active_mib": [
            round(value / MIB, 3)
            for value in sorted(sizes, reverse=True)[:12]
        ],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "direct",
            "direct-early",
            "fixed",
            "fixed-early",
            "encoder",
        ),
        required=True,
    )

    parser.add_argument(
        "--cycles", type=int, default=8
    )

    parser.add_argument(
        "--trace", action="store_true"
    )

    args = parser.parse_args()

    if args.cycles < 2:
        parser.error("At least two cycles required")

    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")

    torch.manual_seed(42)

    model = TinyModel().cuda().eval()
    encoder = None

    if args.mode == "encoder":
        encoder = OpenQuantEncoder(
            model=model,
            tokenizer=TinyTokenizer(),
            batch_size=4,
            max_length=64,
            max_graphs=1,
            max_graph_cache_mb=128,
            min_capture_free_mb=0,
            graph_capture_max_attempts=0,
        )

    with torch.inference_mode():
        tokens = make_tokens(16)
        encode_sentences(model, tokens)
        del tokens

    cleanup()

    report = {
        "mode": args.mode,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "baseline": memory(),
        "cycles": [],
    }

    if args.trace:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            stacks="python",
            max_entries=12000,
        )

    try:
        for cycle in range(1, args.cycles + 1):
            torch.cuda.reset_peak_memory_stats()

            if args.mode == "encoder":
                run_encoder(encoder)
            else:
                for length in LENGTHS:
                    if args.mode.startswith("direct"):
                        direct_capture(
                            model,
                            length,
                            early_release=(
                                args.mode == "direct-early"
                            ),
                        )
                    else:
                        fixed_capture(
                            model,
                            length,
                            early_release=(
                                args.mode == "fixed-early"
                            ),
                        )

            cleanup()
            current = memory()

            current["cycle"] = cycle
            current["peak_mib"] = round(
                torch.cuda.max_memory_allocated() / MIB,
                3,
            )

            report["cycles"].append(current)

            print(
                f"Cycle {cycle}: "
                f"allocated={current['allocated_mib']:.3f} MiB, "
                f"reserved={current['reserved_mib']:.3f} MiB, "
                f"peak={current['peak_mib']:.3f} MiB"
            )

        readings = [
            row["allocated_mib"]
            for row in report["cycles"]
        ]

        report["drift_mib"] = round(
            readings[-1] - readings[0],
            3,
        )

        report["allocator"] = allocator_state()

        Path("reports").mkdir(
            parents=True,
            exist_ok=True,
        )

        output = Path(
            f"reports/graph_layers_{args.mode}.json"
        )

        output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        if args.trace:
            trace_path = Path(
                f"reports/graph_layers_{args.mode}.pickle"
            )

            torch.cuda.memory._dump_snapshot(
                str(trace_path)
            )

            print("Memory trace:", trace_path.resolve())

        print()
        print("Correctness: PASS")
        print(
            "First-to-last allocated drift:",
            report["drift_mib"],
            "MiB",
        )
        print(
            "Largest active blocks:",
            report["allocator"]["largest_active_mib"],
        )
        print("Report:", output.resolve())

    finally:
        if args.trace:
            torch.cuda.memory._record_memory_history(
                enabled=None
            )


if __name__ == "__main__":
    main()
