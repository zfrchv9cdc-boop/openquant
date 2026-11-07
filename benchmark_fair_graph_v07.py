
"""OpenQuant v0.7 - fair, multi-workload CUDA Graph benchmark.

Examples:
    python benchmark_fair_graph_v07.py
    python benchmark_fair_graph_v07.py --all
    python benchmark_fair_graph_v07.py --batch 12 --seq-length 128

Compares:
    FP32 eager / graph
    FP16 eager / graph
    INT4 eager / graph

Also checks:
    numerical correctness
    changed-input correctness
    allocated GPU memory
    embedding fidelity
"""

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from benchmark_graph_v07 import (
    GraphedSentenceEncoder,
    SENTENCES,
)
from openquant.inference_v07 import (
    MODEL_ID,
    encode_sentences,
    load_fp32_reference,
    load_packed_model,
)


SOURCE = Path("models/minilm/model.safetensors")
ARCHIVE = Path("models/minilm/minilm_int4_g32.safetensors")
REPORT_DIR = Path("reports")


def synchronize():
    torch.cuda.synchronize()


def measure(fn, repetitions):
    """Host-observed latency, including output-buffer cloning."""
    measurements = []

    with torch.inference_mode():
        for _ in range(8):
            fn()

        synchronize()

        for _ in range(repetitions):
            synchronize()
            start = time.perf_counter()

            fn()

            synchronize()

            measurements.append(
                (time.perf_counter() - start) * 1000
            )

    return {
        "median_ms": float(median(measurements)),
        "minimum_ms": float(min(measurements)),
        "maximum_ms": float(max(measurements)),
        "samples_ms": measurements,
    }


def load_precision(precision):
    if precision == "INT4":
        model, counts = load_packed_model(ARCHIVE)
        return model.cuda().eval(), counts

    model = load_fp32_reference(SOURCE)

    if precision == "FP16":
        model = model.half()

    return model.cuda().eval(), None


def make_changed_inputs(tokens, vocab_size):
    changed = {
        name: value.clone()
        for name, value in tokens.items()
    }

    original = int(changed["input_ids"][0, 1])
    changed["input_ids"][0, 1] = (
        original + 1
    ) % vocab_size

    return changed


def compare_embeddings(reference, candidate):
    reference = reference.float()
    candidate = candidate.float()

    cosines = F.cosine_similarity(
        reference, candidate, dim=-1
    )

    result = {
        "mean_cosine": float(
            cosines.mean().item()
        ),
        "minimum_cosine": float(
            cosines.min().item()
        ),
    }

    if reference.shape[0] > 1:
        ref_similarity = reference @ reference.T
        new_similarity = candidate @ candidate.T

        ref_similarity.fill_diagonal_(-float("inf"))
        new_similarity.fill_diagonal_(-float("inf"))

        agreement = (
            ref_similarity.argmax(dim=1)
            == new_similarity.argmax(dim=1)
        ).float().mean()

        result["nearest_neighbor_agreement"] = (
            float(agreement.item())
        )

    return result


def measure_precision(
    precision,
    tokens,
    vocab_size,
    repetitions,
):
    gc.collect()
    torch.cuda.empty_cache()
    synchronize()

    base_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    model, counts = load_precision(precision)

    with torch.inference_mode():
        expected = encode_sentences(
            model, tokens
        ).detach().clone()

    # This CPU result is retained for comparing precisions.
    embeddings = expected.float().cpu()

    eager = measure(
        lambda: encode_sentences(model, tokens),
        repetitions,
    )

    eager_allocated = (
        torch.cuda.memory_allocated()
        - base_allocated
    )

    record = {
        "precision": precision,
        "compressed_modules": counts,
        "eager": eager,
        "eager_allocated_bytes": eager_allocated,
    }

    print(
        f"  Eager: {eager['median_ms']:.4f} ms",
        flush=True,
    )

    graph = None

    try:
        graph = GraphedSentenceEncoder(
            model, tokens
        )

        with torch.inference_mode():
            actual = graph(tokens).detach().clone()

        synchronize()

        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=0.001,
            atol=0.001,
        )

        # The captured computation must respond to new IDs.
        changed = make_changed_inputs(
            tokens, vocab_size
        )

        with torch.inference_mode():
            changed_reference = encode_sentences(
                model, changed
            ).detach().clone()

            changed_actual = graph(
                changed
            ).detach().clone()

        synchronize()

        torch.testing.assert_close(
            changed_actual.float(),
            changed_reference.float(),
            rtol=0.001,
            atol=0.001,
        )

        # Restore the original inputs.
        with torch.inference_mode():
            graph(tokens)

        synchronize()

        # Clone because the graph's output buffer is reused.
        # This makes the result independently consumable.
        graph_result = measure(
            lambda: graph(tokens).clone(),
            repetitions,
        )

        graph_allocated = (
            torch.cuda.memory_allocated()
            - base_allocated
        )

        record.update({
            "graph": graph_result,
            "graph_allocated_bytes": graph_allocated,
            "graph_extra_allocated_bytes": (
                graph_allocated - eager_allocated
            ),
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated()
                - base_allocated
            ),
            "graph_speedup": (
                eager["median_ms"]
                / graph_result["median_ms"]
            ),
            "graph_latency_reduction_pct": (
                100 * (
                    1 - graph_result["median_ms"]
                    / eager["median_ms"]
                )
            ),
            "correctness": "PASS",
        })

        print(
            f"  Graph: "
            f"{graph_result['median_ms']:.4f} ms",
            flush=True,
        )

        print(
            f"  Graph speedup: "
            f"{record['graph_speedup']:.2f}x",
            flush=True,
        )

        print("  Correctness: PASS", flush=True)

    except Exception as exc:
        record["correctness"] = "GRAPH_FAILED"
        record["graph_error"] = repr(exc)

        print(
            "  Graph failure:", repr(exc),
            flush=True,
        )

    finally:
        synchronize()

    return record, embeddings


def make_tokens(tokenizer, batch, seq_length):
    sentences = SENTENCES[:batch]

    return tokenizer(
        sentences,
        padding=(
            "max_length" if seq_length else True
        ),
        truncation=True,
        max_length=seq_length or 256,
        return_tensors="pt",
    )


def run_workload(
    tokenizer,
    batch,
    seq_length,
    repetitions,
):
    cpu_tokens = make_tokens(
        tokenizer, batch, seq_length
    )

    tokens = {
        name: value.cuda()
        for name, value in cpu_tokens.items()
    }

    actual_seq = int(
        tokens["input_ids"].shape[1]
    )

    print("\n" + "=" * 70)
    print(
        f"WORKLOAD: batch={batch}, sequence={actual_seq}"
    )
    print("=" * 70)

    report = {
        "batch_size": batch,
        "sequence_length": actual_seq,
        "precisions": {},
        "quality": {},
    }

    embeddings = {}

    for precision in ("FP32", "FP16", "INT4"):
        print(f"\nTesting {precision}...", flush=True)

        record, output = measure_precision(
            precision,
            tokens,
            tokenizer.vocab_size,
            repetitions,
        )

        report["precisions"][precision] = record
        embeddings[precision] = output

        # The model and its graph are no longer referenced
        # by the returned data. Release unused allocations.
        gc.collect()
        torch.cuda.empty_cache()

    reference = embeddings["FP32"]

    for precision in ("FP16", "INT4"):
        report["quality"][precision] = (
            compare_embeddings(
                reference,
                embeddings[precision],
            )
        )

    rows = report["precisions"]

    print("\nFAIR RESULTS")
    print("-" * 70)

    for name in ("FP32", "FP16", "INT4"):
        row = rows[name]

        if "graph" in row:
            print(
                f"{name:<5} "
                f"eager={row['eager']['median_ms']:.4f} ms "
                f"graph={row['graph']['median_ms']:.4f} ms"
            )
        else:
            print(f"{name:<5} GRAPH FAILED")

    if all("graph" in rows[p] for p in rows):
        fp32_ms = rows["FP32"]["graph"]["median_ms"]
        fp16_ms = rows["FP16"]["graph"]["median_ms"]
        int4_ms = rows["INT4"]["graph"]["median_ms"]

        report["comparisons"] = {
            "int4_speedup_vs_fp32_graph": (
                fp32_ms / int4_ms
            ),
            "int4_speedup_vs_fp16_graph": (
                fp16_ms / int4_ms
            ),
            "int4_latency_reduction_vs_fp16_pct": (
                100 * (1 - int4_ms / fp16_ms)
            ),
        }

        if int4_ms < min(fp16_ms, fp32_ms):
            report["decision"] = "INT4_GRAPH_FASTEST"
        else:
            report["decision"] = "MEMORY_SPEED_TRADEOFF"

    else:
        report["decision"] = "INCOMPLETE"

    print("\nDECISION:", report["decision"])

    print(
        "INT4 embedding cosine:",
        f"{report['quality']['INT4']['mean_cosine']:.6f}",
    )

    del embeddings
    del tokens

    gc.collect()
    torch.cuda.empty_cache()

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch", type=int, default=12
    )
    parser.add_argument(
        "--seq-length", type=int, default=0
    )
    parser.add_argument(
        "--repetitions", type=int, default=30
    )
    parser.add_argument(
        "--all", action="store_true"
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")

    if not 1 <= args.batch <= len(SENTENCES):
        parser.error("Batch must be 1..12")

    if (
        args.seq_length != 0
        and not 4 <= args.seq_length <= 512
    ):
        parser.error("Sequence length must be 4..512")

    if args.repetitions < 5:
        parser.error("At least five repetitions required")

    if not SOURCE.is_file() or not ARCHIVE.is_file():
        raise FileNotFoundError(
            "Original or compressed checkpoint missing"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    scenarios = (
        [
            (1, 32),
            (4, 64),
            (12, 12),
            (12, 128),
        ]
        if args.all
        else [(args.batch, args.seq_length)]
    )

    report = {
        "project": "OpenQuant",
        "phase": "v0.7-fair-graph",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "repetitions": args.repetitions,
        "scope": (
            "Pretokenized GPU-resident inputs; "
            "host-observed synchronized forward latency. "
            "Graph timing includes copying fixed inputs "
            "and cloning the result. "
            "Model load and tokenization excluded."
        ),
        "workloads": [],
    }

    for batch, seq in scenarios:
        report["workloads"].append(
            run_workload(
                tokenizer,
                batch,
                seq,
                args.repetitions,
            )
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = REPORT_DIR / "fair_graph_v07.json"

    path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nREPORT:", path.resolve())

    completed = all(
        workload["decision"] != "INCOMPLETE"
        for workload in report["workloads"]
    )

    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
