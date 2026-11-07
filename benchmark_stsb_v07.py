
"""Evaluate real MiniLM FP32 and OpenQuant INT4 on STS-B.

Run:
    python benchmark_stsb_v07.py --limit 256
    python benchmark_stsb_v07.py --limit 0

Zero means the complete validation split.
"""

import argparse
import gc
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from openquant.encoder_v07 import OpenQuantEncoder
from openquant.inference_v07 import (
    MODEL_ID,
    encode_sentences,
    load_fp32_reference,
)
from openquant.quality_metrics_v07 import evaluate_similarity


SOURCE = Path("models/minilm/model.safetensors")
ARCHIVE = Path("models/minilm/minilm_int4_g32.safetensors")
OUTPUT = Path("reports/stsb_quality_v07.json")


def encode_fp32(model, tokenizer, texts, batch_size, max_length):
    outputs = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]

            tokens = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            tokens = {
                key: value.cuda()
                for key, value in tokens.items()
            }

            embeddings = encode_sentences(model, tokens)
            outputs.append(embeddings.float().cpu())

            if (start // batch_size) % 25 == 0:
                print(
                    f"FP32: {min(start + batch_size, len(texts))}"
                    f"/{len(texts)}",
                    flush=True,
                )

    return torch.cat(outputs, dim=0)


def pair_scores(embeddings, left, right):
    a = embeddings[left]
    b = embeddings[right]

    return F.cosine_similarity(a, b, dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)

    args = parser.parse_args()

    if args.limit < 0 or args.batch < 1:
        parser.error("Invalid evaluation settings")

    if not 4 <= args.max_length <= 512:
        parser.error("max-length must be between 4 and 512")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    if not SOURCE.is_file() or not ARCHIVE.is_file():
        raise FileNotFoundError("Model checkpoints missing")

    print("OPENQUANT v0.7 — LABELED STS QUALITY EVALUATION")
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

    gold = [float(row["score"]) for row in rows]
    left_sentences = [row["sentence1"] for row in rows]
    right_sentences = [row["sentence2"] for row in rows]

    # Encode each unique sentence once.
    texts = list(dict.fromkeys(
        left_sentences + right_sentences
    ))

    text_index = {
        sentence: index
        for index, sentence in enumerate(texts)
    }

    left = [
        text_index[sentence]
        for sentence in left_sentences
    ]

    right = [
        text_index[sentence]
        for sentence in right_sentences
    ]

    print("Labeled pairs:", len(rows))
    print("Unique sentences:", len(texts))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    # Evaluate FP32 reference first.
    reference_model = (
        load_fp32_reference(SOURCE)
        .cuda()
        .eval()
    )

    fp32_embeddings = encode_fp32(
        reference_model,
        tokenizer,
        texts,
        args.batch,
        args.max_length,
    )

    del reference_model
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate INT4 using the same tokenizer and length limit.
    int4_encoder = OpenQuantEncoder(
        archive=ARCHIVE,
        tokenizer=tokenizer,
        batch_size=args.batch,
        max_length=args.max_length,
        enable_graphs=False,
        max_graphs=0,
    )

    int4_embeddings = int4_encoder.encode(
        texts,
        use_graph=False,
        return_cpu=True,
    ).float()

    fp32_embeddings = fp32_embeddings.float()

    reference_scores = pair_scores(
        fp32_embeddings, left, right
    )

    compressed_scores = pair_scores(
        int4_embeddings, left, right
    )

    metrics = evaluate_similarity(
        gold,
        reference_scores,
        compressed_scores,
    )

    vector_cosine = F.cosine_similarity(
        fp32_embeddings,
        int4_embeddings,
        dim=1,
    )

    metrics["mean_embedding_cosine"] = float(
        vector_cosine.mean().item()
    )

    metrics["minimum_embedding_cosine"] = float(
        vector_cosine.min().item()
    )

    report = {
        "project": "OpenQuant",
        "phase": "v0.7_quality",
        "dataset": "sentence-transformers/stsb",
        "split": "validation",
        "dataset_fingerprint": getattr(
            dataset, "_fingerprint", None
        ),
        "seed": args.seed,
        "limit": args.limit,
        "batch_size": args.batch,
        "max_length": args.max_length,
        "fp32_source": str(SOURCE),
        "int4_source": str(ARCHIVE),
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

    print("\nQUALITY RESULTS")
    print("-" * 60)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    print("\nREPORT:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
