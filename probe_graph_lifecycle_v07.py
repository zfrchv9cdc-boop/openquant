
"""Isolate CUDA Graph memory growth across three workloads."""

import argparse
import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from openquant.encoder_v07 import OpenQuantEncoder

MIB = 1024 ** 2


class TinyTokenizer:
    pad_token_id = 0

    def __call__(
        self, texts, padding=True, truncation=True,
        max_length=64, return_tensors="pt",
    ):
        sequences = [
            ([1] + [4 + ord(c) % 60 for c in text] + [2])[
                :max_length
            ]
            for text in texts
        ]
        width = max(len(s) for s in sequences)
        ids = torch.zeros(
            len(texts), width, dtype=torch.long
        )
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1
        return {
            "input_ids": ids,
            "attention_mask": mask,
        }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(80, 16)
        self.linear = nn.Linear(16, 16)

    def forward(self, input_ids, attention_mask):
        x = self.linear(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=x)


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


def cleanup():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def run_minimal():
    x = torch.ones(
        512, 512, device="cuda", dtype=torch.float32
    )

    with torch.inference_mode():
        warmup = (x * 1.01).relu()

    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()

    with torch.inference_mode():
        with torch.cuda.graph(graph):
            output = (x * 1.01).relu()

    graph.replay()
    torch.cuda.synchronize()

    graph.reset()
    del output, graph, warmup, x


def run_encoder(encoder, cases):
    for texts in cases:
        output = encoder.encode(texts, use_graph=True)
        del output

    references = [
        weakref.ref(graph)
        for graph in encoder.graphs.values()
    ]

    encoder.clear_graph_cache()

    return sum(ref() is not None for ref in references)


def allocator_summary():
    try:
        snapshot = torch.cuda.memory_snapshot()
        states = {}
        segments = {}

        for segment in snapshot:
            kind = segment.get("segment_type", "unknown")
            segments[kind] = segments.get(kind, 0) + 1

            for block in segment.get("blocks", []):
                state = block.get("state", "unknown")
                states[state] = (
                    states.get(state, 0)
                    + int(block.get("size", 0))
                )

        return {
            "block_mib": {
                key: round(value / MIB, 3)
                for key, value in states.items()
            },
            "segment_counts": segments,
        }
    except Exception as exc:
        return {"snapshot_error": repr(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("minimal", "tiny", "int4"),
        required=True,
    )
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--cap", type=int, default=0)

    args = parser.parse_args()

    if args.cycles < 2 or args.cap < 0:
        parser.error("Invalid cycles or capture cap")

    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")

    encoder = None
    cases = []

    if args.mode != "minimal":
        options = dict(
            max_graphs=1,
            max_graph_cache_mb=128,
            min_capture_free_mb=0,
            graph_capture_min_calls=1,
        )

        # Compatible with the existing encoder when --cap=0.
        if args.cap:
            options["graph_capture_max_attempts"] = args.cap

        if args.mode == "tiny":
            options.update(
                model=TinyModel(),
                tokenizer=TinyTokenizer(),
                batch_size=4,
                max_length=64,
            )
            cases = [
                ["cat"],
                ["x" * 20],
                ["z" * 40],
                ["cat"],
            ]
        else:
            options.update(
                batch_size=12,
                max_length=128,
            )
            from benchmark_capture_peak_v07 import CASES
            cases = [texts for _, texts in CASES]

        encoder = OpenQuantEncoder(**options)

        encoder.encode(cases[0], use_graph=False)

    cleanup()
    baseline = memory()
    records = []

    print("Mode:", args.mode)
    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Baseline:", baseline)

    for cycle in range(1, args.cycles + 1):
        if args.mode == "minimal":
            run_minimal()
            live = None
        else:
            live = run_encoder(encoder, cases)

        cleanup()
        reading = memory()

        info = (
            encoder.cache_info()
            if encoder is not None else None
        )

        row = {
            "cycle": cycle,
            **reading,
            "live_graph_wrappers": live,
            "capture_attempts": (
                info.get("graph_capture_attempts")
                if info is not None else None
            ),
        }
        records.append(row)

        print(
            f"Cycle {cycle}: "
            f"allocated={reading['allocated_mib']:.3f} MiB; "
            f"reserved={reading['reserved_mib']:.3f} MiB; "
            f"live_wrappers={live}"
        )

    drift = round(
        records[-1]["allocated_mib"]
        - records[0]["allocated_mib"],
        3,
    )

    report = {
        "mode": args.mode,
        "cap": args.cap,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "baseline": baseline,
        "cycles": records,
        "drift_mib": drift,
        "allocator": allocator_summary(),
    }

    suffix = f"_cap{args.cap}" if args.cap else ""
    output = Path(
        f"reports/graph_lifecycle_{args.mode}{suffix}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("First-to-last drift:", drift, "MiB")
    print("Allocator:", report["allocator"])
    print("Report:", output.resolve())


if __name__ == "__main__":
    main()
