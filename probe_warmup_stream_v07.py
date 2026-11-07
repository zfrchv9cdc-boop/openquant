
"""Compare CUDA Graph warmup stream allocation lifetimes."""

import argparse
import gc
import json
from pathlib import Path

import torch

from openquant.inference_v07 import encode_sentences
from probe_graph_lifecycle_v07 import TinyModel

MIB = 1024 ** 2


def tokens_for(length):
    return {
        "input_ids": torch.full(
            (1, length), 7, device="cuda",
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(
            (1, length), device="cuda",
            dtype=torch.long,
        ),
    }


def memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return round(
        torch.cuda.memory_allocated() / MIB, 3
    )


def capture_once(model, length, mode):
    tokens = tokens_for(length)
    static = {
        key: value.clone()
        for key, value in tokens.items()
    }

    with torch.inference_mode():
        reference = encode_sentences(
            model, static
        ).detach().clone()

        if mode == "side":
            current = torch.cuda.current_stream()
            warmup = torch.cuda.Stream()

            warmup.wait_stream(current)

            with torch.cuda.stream(warmup):
                for _ in range(4):
                    encode_sentences(model, static)

            current.wait_stream(warmup)

        elif mode == "current":
            for _ in range(4):
                encode_sentences(model, static)

        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            captured = encode_sentences(
                model, static
            )

        graph.replay()
        actual = captured.clone()

        torch.cuda.synchronize()
        torch.testing.assert_close(
            actual, reference,
            rtol=0.001, atol=0.001,
        )

    graph.reset()

    del actual, reference, captured
    del graph, static, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("none", "current", "side"),
        required=True,
    )
    parser.add_argument(
        "--cycles", type=int, default=8
    )
    args = parser.parse_args()

    if args.cycles < 2:
        parser.error("Use at least two cycles")

    torch.manual_seed(42)
    model = TinyModel().cuda().eval()

    with torch.inference_mode():
        encode_sentences(model, tokens_for(16))

    baseline = memory()
    rows = []

    print("GPU:", torch.cuda.get_device_name(0))
    print("Warmup:", args.mode)
    print("Baseline:", baseline, "MiB")

    for cycle in range(1, args.cycles + 1):
        for length in (16, 32, 64, 16):
            capture_once(model, length, args.mode)

        allocated = memory()
        rows.append(allocated)

        print(
            f"Cycle {cycle}: {allocated:.3f} MiB"
        )

    drift = round(rows[-1] - rows[0], 3)
    report = {
        "mode": args.mode,
        "cycles": rows,
        "baseline_mib": baseline,
        "first_to_last_drift_mib": drift,
    }

    output = Path(
        f"reports/warmup_stream_{args.mode}.json"
    )
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Correctness: PASS")
    print("Memory drift:", drift, "MiB")
    print("Report:", output.resolve())


if __name__ == "__main__":
    main()
