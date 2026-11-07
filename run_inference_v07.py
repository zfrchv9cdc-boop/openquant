
"""Compare FP32 MiniLM with OpenQuant packed INT4 inference."""

import gc
import json
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from openquant.inference_v07 import (
    MODEL_ID,
    encode_sentences,
    load_fp32_reference,
    load_packed_model,
)


SOURCE = Path("models/minilm/model.safetensors")
ARCHIVE = Path("models/minilm/minilm_int4_g32.safetensors")
REPORT = Path("reports/inference_v07.json")

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


def benchmark(model, tokens, repetitions=15):
    """Host-observed forward latency, excluding tokenization."""
    with torch.inference_mode():
        for _ in range(4):
            encode_sentences(model, tokens)

        torch.cuda.synchronize()

        measurements = []

        for _ in range(repetitions):
            torch.cuda.synchronize()
            start = time.perf_counter()

            encode_sentences(model, tokens)

            torch.cuda.synchronize()

            measurements.append(
                (time.perf_counter() - start) * 1000
            )

    return median(measurements)


def evaluate_quality(reference, compressed):
    """Small-sample embedding fidelity diagnostics."""
    reference = reference.float()
    compressed = compressed.float()

    cosines = F.cosine_similarity(
        reference, compressed, dim=-1
    )

    similarities_ref = reference @ reference.T
    similarities_int4 = compressed @ compressed.T

    difference = (
        similarities_ref - similarities_int4
    ).abs()

    # Exclude the self-similarity diagonal.
    ref_neighbors = similarities_ref.clone()
    int4_neighbors = similarities_int4.clone()

    ref_neighbors.fill_diagonal_(-float("inf"))
    int4_neighbors.fill_diagonal_(-float("inf"))

    matching_neighbors = (
        ref_neighbors.argmax(dim=1)
        == int4_neighbors.argmax(dim=1)
    )

    return {
        "mean_embedding_cosine": float(
            cosines.mean().item()
        ),
        "minimum_embedding_cosine": float(
            cosines.min().item()
        ),
        "maximum_similarity_difference": float(
            difference.max().item()
        ),
        "nearest_neighbor_agreement": float(
            matching_neighbors.float().mean().item()
        ),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)

    if not ARCHIVE.with_suffix(".json").is_file():
        raise FileNotFoundError(
            ARCHIVE.with_suffix(".json")
        )

    print("OPENQUANT V0.7 — REAL MODEL INFERENCE")
    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    tokenized = tokenizer(
        SENTENCES,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    tokens = {
        key: value.cuda()
        for key, value in tokenized.items()
    }

    print("Sentences:", len(SENTENCES))
    print("Sequence length:", tokens["input_ids"].shape[1])

    # First run the full-precision model.
    reference_model = (
        load_fp32_reference(SOURCE).cuda().eval()
    )

    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        reference_embeddings = encode_sentences(
            reference_model, tokens
        ).cpu()

    fp32_ms = benchmark(reference_model, tokens)

    fp32_resident = torch.cuda.memory_allocated()
    fp32_peak = torch.cuda.max_memory_allocated()

    print(f"FP32 forward latency: {fp32_ms:.3f} ms")

    del reference_model
    gc.collect()
    torch.cuda.empty_cache()

    # Now run the packed INT4 model.
    compressed_model, counts = load_packed_model(
        ARCHIVE
    )

    compressed_model = compressed_model.cuda().eval()

    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        compressed_embeddings = encode_sentences(
            compressed_model, tokens
        ).cpu()

    int4_ms = benchmark(compressed_model, tokens)

    int4_resident = torch.cuda.memory_allocated()
    int4_peak = torch.cuda.max_memory_allocated()

    quality = evaluate_quality(
        reference_embeddings,
        compressed_embeddings,
    )

    speedup = fp32_ms / int4_ms

    print("\nRESULTS")
    print("-" * 60)

    print("Compressed Linear layers:", counts["linear"])
    print("Compressed Embedding tables:", counts["embedding"])

    print(f"FP32 latency: {fp32_ms:.3f} ms")
    print(f"INT4 latency: {int4_ms:.3f} ms")
    print(f"FP32/INT4 speed ratio: {speedup:.3f}x")

    print(
        "Mean embedding cosine:",
        round(quality["mean_embedding_cosine"], 6),
    )

    print(
        "Nearest-neighbor agreement:",
        round(100 * quality["nearest_neighbor_agreement"], 2),
        "%",
    )

    mib = 1024 ** 2

    print(
        f"FP32 resident CUDA memory: "
        f"{fp32_resident / mib:.2f} MiB"
    )

    print(
        f"INT4 resident CUDA memory: "
        f"{int4_resident / mib:.2f} MiB"
    )

    print(
        f"FP32 peak allocated memory: "
        f"{fp32_peak / mib:.2f} MiB"
    )

    print(
        f"INT4 peak allocated memory: "
        f"{int4_peak / mib:.2f} MiB"
    )

    report = {
        "project": "OpenQuant",
        "milestone": "0.7-real-inference",
        "gpu": torch.cuda.get_device_name(0),
        "model": MODEL_ID,
        "sample_count": len(SENTENCES),
        "sequence_length": int(
            tokens["input_ids"].shape[1]
        ),
        "compressed_modules": counts,
        "forward_latency_ms": {
            "fp32": fp32_ms,
            "int4": int4_ms,
            "fp32_over_int4": speedup,
        },
        "memory_bytes": {
            "fp32_resident": fp32_resident,
            "int4_resident": int4_resident,
            "fp32_peak": fp32_peak,
            "int4_peak": int4_peak,
        },
        "fidelity": quality,
        "benchmark_scope": (
            "Model forward and mean pooling with "
            "pretokenized GPU inputs. Excludes model "
            "load, tokenizer execution and CPU-to-GPU "
            "input transfer."
        ),
    }

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport:", REPORT.resolve())


if __name__ == "__main__":
    main()
