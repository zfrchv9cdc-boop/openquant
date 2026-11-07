
"""OpenQuant v0.7 - cached STS-B quality benchmark.

First full run builds an FP32 reference cache.
Subsequent runs reuse verified reference embeddings.

The INT4 model is always evaluated fresh.

Commands:
  python benchmark_stsb_fast_v07.py --limit 128
  python benchmark_stsb_fast_v07.py --limit 0
  python benchmark_stsb_fast_v07.py --limit 0 --refresh-reference
"""

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import transformers

from benchmark_stsb_v07 import (
    encode_fp32,
    pair_scores,
)
from openquant.encoder_v07 import OpenQuantEncoder
from openquant.inference_v07 import (
    MODEL_ID,
    load_fp32_reference,
)
from openquant.quality_metrics_v07 import evaluate_similarity


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "models/minilm/model.safetensors"
ARCHIVE = ROOT / "models/minilm/minilm_int4_g32.safetensors"
CACHE_DIR = ROOT / "reports/cache"
OUTPUT = ROOT / "reports/stsb_fast_v07.json"


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as stream:
        for chunk in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def reference_key(
    tokenizer,
    texts,
    dataset_fingerprint,
    batch_size,
    max_length,
):
    """Invalidate cache when reference inputs or code change."""

    backend = tokenizer.backend_tokenizer.to_str()

    metadata = {
        "schema": 1,
        "model_id": MODEL_ID,
        "checkpoint_sha256": sha256_file(SOURCE),
        "inference_code_sha256": sha256_file(
            ROOT / "openquant/inference_v07.py"
        ),
        "reference_code_sha256": sha256_file(
            ROOT / "benchmark_stsb_v07.py"
        ),
        "tokenizer_sha256": sha256_text(backend),
        "texts_sha256": sha256_text(
            json.dumps(
                texts,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
        "dataset_fingerprint": dataset_fingerprint,
        "batch_size": batch_size,
        "max_length": max_length,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

    key = sha256_text(
        json.dumps(
            metadata,
            sort_keys=True,
            default=str,
        )
    )[:24]

    return key, metadata


def load_reference(
    tokenizer,
    texts,
    args,
    dataset_fingerprint,
):
    key, metadata = reference_key(
        tokenizer,
        texts,
        dataset_fingerprint,
        args.batch,
        args.max_length,
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = CACHE_DIR / f"stsb_fp32_{key}.npy"

    if path.exists() and not args.refresh_reference:
        try:
            array = np.load(
                path,
                allow_pickle=False,
            )

            if (
                array.dtype != np.float32
                or array.ndim != 2
                or array.shape[0] != len(texts)
                or array.shape[1] != 384
                or not np.isfinite(array).all()
            ):
                raise ValueError(
                    "Invalid cached reference embeddings"
                )

            print("FP32 reference cache: HIT")
            return (
                torch.from_numpy(array.copy()),
                path,
                True,
                metadata,
            )

        except (ValueError, OSError) as exc:
            print(
                "Invalid cache; rebuilding:",
                str(exc),
            )

    print("FP32 reference cache: MISS")

    reference_model = (
        load_fp32_reference(SOURCE)
        .cuda()
        .eval()
    )

    reference = encode_fp32(
        reference_model,
        tokenizer,
        texts,
        args.batch,
        args.max_length,
    ).float().cpu()

    torch.cuda.synchronize()

    del reference_model
    gc.collect()
    torch.cuda.empty_cache()

    temporary = path.with_suffix(".tmp.npy")

    np.save(
        temporary,
        reference.numpy(),
        allow_pickle=False,
    )

    temporary.replace(path)

    print("FP32 reference cached:", path)

    return reference, path, False, metadata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--refresh-reference",
        action="store_true",
    )

    args = parser.parse_args()

    if args.limit < 0 or args.limit in (1, 2):
        parser.error("Use 0 for full evaluation or at least 3")

    if not 1 <= args.batch <= 128:
        parser.error("Batch size must be 1..128")

    if not 4 <= args.max_length <= 512:
        parser.error("Invalid maximum length")

    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")

    if not SOURCE.is_file() or not ARCHIVE.is_file():
        parser.error("Checkpoint files missing")

    started = time.perf_counter()

    print("OPENQUANT FAST STS-B EVALUATION")
    print("GPU:", torch.cuda.get_device_name(0))

    dataset = load_dataset(
        "sentence-transformers/stsb",
        split="validation",
    )

    indices = list(range(len(dataset)))

    if args.limit and args.limit < len(indices):
        rng = random.Random(args.seed)
        indices = sorted(
            rng.sample(indices, args.limit)
        )

    rows = [dataset[i] for i in indices]

    left_sentences = [
        row["sentence1"] for row in rows
    ]

    right_sentences = [
        row["sentence2"] for row in rows
    ]

    gold = [
        float(row["score"]) for row in rows
    ]

    texts = list(dict.fromkeys(
        left_sentences + right_sentences
    ))

    lookup = {
        sentence: index
        for index, sentence in enumerate(texts)
    }

    left_indices = [
        lookup[value] for value in left_sentences
    ]

    right_indices = [
        lookup[value] for value in right_sentences
    ]

    print("Pairs:", len(rows))
    print("Unique sentences:", len(texts))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    # Stage 1: cached FP32 reference.
    t0 = time.perf_counter()

    reference, cache_path, cache_hit, metadata = (
        load_reference(
            tokenizer,
            texts,
            args,
            getattr(dataset, "_fingerprint", None),
        )
    )

    reference_seconds = time.perf_counter() - t0

    # Stage 2: always recompute INT4.
    t0 = time.perf_counter()

    encoder = OpenQuantEncoder(
        archive=ARCHIVE,
        tokenizer=tokenizer,
        batch_size=args.batch,
        max_length=args.max_length,
        enable_graphs=False,
        max_graphs=0,
    )

    compressed = encoder.encode(
        texts,
        use_graph=False,
        return_cpu=True,
    ).float()

    torch.cuda.synchronize()

    int4_seconds = time.perf_counter() - t0

    # Stage 3: compare the same labeled pairs.
    reference_scores = pair_scores(
        reference,
        left_indices,
        right_indices,
    )

    compressed_scores = pair_scores(
        compressed,
        left_indices,
        right_indices,
    )

    metrics = evaluate_similarity(
        gold,
        reference_scores,
        compressed_scores,
    )

    cosine = F.cosine_similarity(
        reference,
        compressed,
        dim=1,
    )

    metrics["mean_embedding_cosine"] = float(
        cosine.mean().item()
    )

    metrics["minimum_embedding_cosine"] = float(
        cosine.min().item()
    )

    total_seconds = time.perf_counter() - started

    report = {
        "project": "OpenQuant",
        "phase": "v0.7_cached_stsb",
        "dataset": "sentence-transformers/stsb",
        "split": "validation",
        "pairs": len(rows),
        "unique_sentences": len(texts),
        "batch_size": args.batch,
        "max_length": args.max_length,
        "fp32_cache_hit": cache_hit,
        "fp32_cache_path": str(cache_path),
        "reference_metadata": metadata,
        "timing_seconds": {
            "reference_stage": reference_seconds,
            "int4_stage": int4_seconds,
            "total": total_seconds,
        },
        "metrics": metrics,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nQUALITY")
    print("-" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\nTIMING")
    print(f"FP32 reference stage: {reference_seconds:.3f} s")
    print(f"INT4 evaluation stage: {int4_seconds:.3f} s")
    print(f"Total: {total_seconds:.3f} s")
    print(f"Reference cache hit: {cache_hit}")
    print("Report:", OUTPUT)


if __name__ == "__main__":
    main()
