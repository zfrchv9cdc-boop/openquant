
"""OpenQuant v0.7: benchmark eager vs CUDA Graph inference.

Tests actual full-model execution, including pooling.
Preserves the existing inference implementation.

Run:
    python benchmark_graph_v07.py
"""

import gc
import json
import time
from pathlib import Path
from statistics import median

import torch
from transformers import AutoTokenizer

from openquant.inference_v07 import (
    MODEL_ID,
    encode_sentences,
    load_fp32_reference,
    load_packed_model,
)


SOURCE = Path("models/minilm/model.safetensors")
ARCHIVE = Path("models/minilm/minilm_int4_g32.safetensors")
REPORT = Path("reports/inference_graph_v07.json")

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


def measure(callable_fn, repetitions=30):
    """Measure host-observed latency with per-call synchronization."""

    with torch.inference_mode():
        for _ in range(6):
            callable_fn()

    torch.cuda.synchronize()

    values = []

    for _ in range(repetitions):
        torch.cuda.synchronize()

        started = time.perf_counter()

        with torch.inference_mode():
            callable_fn()

        torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        values.append(elapsed_ms)

    return {
        "median_ms": median(values),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "samples_ms": values,
    }


class GraphedSentenceEncoder:
    """Capture a fixed-shape sentence encoder for repeated inference."""

    def __init__(self, model, example_tokens):
        self.model = model.eval()

        self.keys = tuple(sorted(example_tokens))

        # Graph inputs must retain stable GPU memory addresses.
        self.static_inputs = {
            key: example_tokens[key].clone()
            for key in self.keys
        }

        current = torch.cuda.current_stream()

        # Warmup on a separate CUDA stream before capture.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(current)

        with torch.cuda.stream(warmup_stream):
            with torch.inference_mode():
                for _ in range(5):
                    encode_sentences(
                        self.model,
                        self.static_inputs,
                    )

        current.wait_stream(warmup_stream)

        self.graph = torch.cuda.CUDAGraph()

        # Capture model forward AND sentence pooling.
        with torch.inference_mode():
            with torch.cuda.graph(self.graph):
                self.static_output = encode_sentences(
                    self.model,
                    self.static_inputs,
                )

        torch.cuda.synchronize()

    def __call__(self, tokens):
        if tuple(sorted(tokens)) != self.keys:
            raise ValueError(
                "Input keys differ from the captured graph"
            )

        for key in self.keys:
            source = tokens[key]
            target = self.static_inputs[key]

            if (
                source.shape != target.shape
                or source.dtype != target.dtype
                or source.device != target.device
            ):
                raise ValueError(
                    "CUDA Graph inputs must keep their "
                    "captured shape, dtype and device"
                )

            target.copy_(source)

        self.graph.replay()

        # This buffer is reused by every replay.
        # Clone it if the caller needs a persistent result.
        return self.static_output


def print_latency(name, result):
    print(
        f"{name:<25}"
        f"{result['median_ms']:>10.3f} ms"
    )


def release_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    if not SOURCE.is_file() or not ARCHIVE.is_file():
        raise FileNotFoundError(
            "Original or INT4 MiniLM checkpoint missing"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    encoded = tokenizer(
        SENTENCES,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    tokens = {
        key: value.cuda()
        for key, value in encoded.items()
    }

    print("=" * 65)
    print("OPENQUANT v0.7 — CUDA GRAPH INFERENCE EXPERIMENT")
    print("=" * 65)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Sentences:", tokens["input_ids"].shape[0])
    print("Sequence length:", tokens["input_ids"].shape[1])

    results = {
        "project": "OpenQuant",
        "experiment": "v0.7_cudagraph_inference",
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": int(tokens["input_ids"].shape[0]),
        "sequence_length": int(tokens["input_ids"].shape[1]),
    }

    # ------------------------------------------------------
    # FP32 baseline.
    # ------------------------------------------------------

    print("\nLoading FP32 baseline...")

    fp32 = load_fp32_reference(SOURCE).cuda().eval()

    fp32_latency = measure(
        lambda: encode_sentences(fp32, tokens)
    )

    results["fp32_eager"] = fp32_latency

    print_latency("FP32 eager", fp32_latency)

    release_model(fp32)
    del fp32

    # ------------------------------------------------------
    # FP16 baseline.
    # ------------------------------------------------------

    print("\nLoading FP16 baseline...")

    fp16 = (
        load_fp32_reference(SOURCE)
        .half()
        .cuda()
        .eval()
    )

    fp16_latency = measure(
        lambda: encode_sentences(fp16, tokens)
    )

    results["fp16_eager"] = fp16_latency

    print_latency("FP16 eager", fp16_latency)

    release_model(fp16)
    del fp16

    # ------------------------------------------------------
    # Packed INT4 eager baseline.
    # ------------------------------------------------------

    print("\nLoading packed INT4 model...")

    int4, counts = load_packed_model(ARCHIVE)
    int4 = int4.cuda().eval()

    results["compressed_modules"] = counts

    with torch.inference_mode():
        eager_reference = encode_sentences(
            int4, tokens
        ).detach().clone()

    eager_latency = measure(
        lambda: encode_sentences(int4, tokens)
    )

    results["int4_eager"] = eager_latency

    print_latency("INT4 eager", eager_latency)

    eager_resident_bytes = torch.cuda.memory_allocated()

    # ------------------------------------------------------
    # Capture full INT4 inference.
    # ------------------------------------------------------

    print("\nCapturing CUDA Graph...")

    try:
        graphed = GraphedSentenceEncoder(
            int4, tokens
        )

        with torch.inference_mode():
            graph_output = graphed(tokens).detach().clone()

        torch.cuda.synchronize()

        torch.testing.assert_close(
            graph_output.float(),
            eager_reference.float(),
            rtol=0.001,
            atol=0.001,
        )

        # Test that changing the input changes the
        # captured model's result correctly.
        alternative = {
            key: value.clone()
            for key, value in tokens.items()
        }

        alternative["input_ids"][0, 2] = (
            alternative["input_ids"][1, 2]
        )

        with torch.inference_mode():
            alternative_eager = encode_sentences(
                int4, alternative
            ).detach().clone()

            alternative_graph = graphed(
                alternative
            ).detach().clone()

        torch.cuda.synchronize()

        torch.testing.assert_close(
            alternative_graph.float(),
            alternative_eager.float(),
            rtol=0.001,
            atol=0.001,
        )

        print("CUDA Graph correctness: PASS")
        print("Changed-input correctness: PASS")

        # Restore original inputs for equivalent benchmarks.
        graphed(tokens)
        torch.cuda.synchronize()

        graph_latency = measure(
            lambda: graphed(tokens)
        )

        results["int4_graph"] = graph_latency

        graph_resident_bytes = (
            torch.cuda.memory_allocated()
        )

        results["memory_bytes"] = {
            "int4_eager_resident": eager_resident_bytes,
            "int4_graph_resident": graph_resident_bytes,
        }

        print_latency("INT4 CUDA Graph", graph_latency)

        eager_ms = eager_latency["median_ms"]
        graph_ms = graph_latency["median_ms"]

        speedup = eager_ms / graph_ms

        reduction = (
            1 - graph_ms / eager_ms
        ) * 100

        results["graph_speedup"] = speedup
        results["graph_latency_reduction_pct"] = reduction

        print("\nRESULTS")
        print("-" * 65)

        print(f"Eager INT4: {eager_ms:.3f} ms")
        print(f"Graphed INT4: {graph_ms:.3f} ms")
        print(f"Speedup: {speedup:.3f}x")
        print(f"Latency reduction: {reduction:.2f}%")

        print(
            f"Graph vs FP32: "
            f"{fp32_latency['median_ms'] / graph_ms:.3f}x"
        )

        print(
            f"Graph vs FP16: "
            f"{fp16_latency['median_ms'] / graph_ms:.3f}x"
        )

        if reduction >= 30:
            decision = "KEEP_GRAPH"
        else:
            decision = "NEXT_BOTTLENECK"

        results["decision"] = decision

        print("\nDECISION:", decision)

    except Exception as exc:
        results["decision"] = "CAPTURE_FAILED"
        results["error"] = repr(exc)

        print("\nCUDA Graph capture or validation failed:")
        print(repr(exc))
        print("The original eager inference code was not modified.")

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSaved:", REPORT.resolve())


if __name__ == "__main__":
    main()
