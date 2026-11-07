"""OpenQuant v0.6 - two-stage, correctness-gated GPU autotuner."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import torch
import triton
import triton.testing

from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)
from openquant.model_eval import load_layers
from openquant.tune_next import candidates


# Candidates chosen from measured RTX 5060 Ti performance.
# These are starting points, not universal optimal settings.
QUICK_CONFIGS = {
    1: {
        "legacy",
        "gemv_original",
        "batched_BN4",
        "packed_default",
    },
    4: {
        "legacy",
        "gemv_original",
        "batched_BN16",
        "packed_32_64_64_4",
    },
    16: {
        "legacy",
        "packed_default",
        "packed_32_64_64_4",
        "packed_64_64_128_4",
    },
    128: {
        "legacy",
        "packed_default",
        "packed_64_64_64_4",
        "packed_128_64_64_4",
    },
}


def benchmark(fn, rep_ms):
    return float(
        triton.testing.do_bench_cudagraph(
            fn,
            rep=rep_ms,
            return_mode="median",
        )
    )


def validate(fn, reference):
    output = fn()
    torch.cuda.synchronize()

    if output.shape != reference.shape:
        raise ValueError("Incorrect output shape")

    if not torch.isfinite(output).all().item():
        raise ValueError("Non-finite output")

    torch.testing.assert_close(
        output.float(),
        reference.float(),
        rtol=0.02,
        atol=0.02,
    )


def incumbent_for_batch(batch):
    if batch <= 4:
        return "gemv_original"
    return "packed_default"


def allowed_names(batch):
    if batch in QUICK_CONFIGS:
        return QUICK_CONFIGS[batch]

    names = {
        "legacy",
        "packed_default",
        "packed_32_64_64_4",
    }
    if batch <= 4:
        names.add("gemv_original")
    return names


def tune_batch(
    batch,
    x,
    packed,
    scales,
    out_features,
    group_size,
    reference,
    args,
):
    candidate_list = candidates(
        batch,
        x,
        packed,
        scales,
        out_features,
        group_size,
    )

    shortlist = allowed_names(batch)
    valid = {}
    failed = []
    screened = []

    print(f"\nBATCH M={batch}")
    print("=" * 64)

    # Stage 1: correctness and inexpensive screening.
    for name, fn, config in candidate_list:
        if args.mode == "quick" and name not in shortlist:
            continue

        try:
            validate(fn, reference)
            latency = benchmark(fn, args.screen_ms)

            if not np.isfinite(latency) or latency <= 0:
                raise ValueError("Invalid benchmark latency")

            valid[name] = (fn, config)
            screened.append({
                "name": name,
                "screen_latency_ms": latency,
            })

            print(f"SCREEN {name:<27} {latency * 1000:.3f} us")

        except Exception as exc:
            failed.append({
                "name": name,
                "reason": str(exc)[:400],
            })
            print(f"FAILED {name}: {str(exc)[:120]}")

    if not screened:
        return {
            "batch_size": batch,
            "best": None,
            "passing": [],
            "failed": failed,
        }

    screened.sort(key=lambda item: item["screen_latency_ms"])

    # Stage 2: retain the top two plus safety baselines.
    finalists = {
        item["name"] for item in screened[:2]
    }

    incumbent = incumbent_for_batch(batch)

    for name in ("legacy", incumbent):
        if name in valid:
            finalists.add(name)

    final_results = []

    for name in sorted(finalists):
        fn, config = valid[name]

        try:
            samples = [
                benchmark(fn, args.final_ms)
                for _ in range(args.final_repeats)
            ]

            latency = median(samples)

            result = {
                "name": name,
                "latency_ms": latency,
                "samples_ms": samples,
                "config": config,
            }

            final_results.append(result)

            print(
                f"FINAL  {name:<27} "
                f"{latency * 1000:.3f} us"
            )

        except Exception as exc:
            failed.append({
                "name": name,
                "reason": "Final benchmark: " + str(exc)[:300],
            })

    if not final_results:
        return {
            "batch_size": batch,
            "best": None,
            "passing": screened,
            "failed": failed,
        }

    final_results.sort(key=lambda item: item["latency_ms"])
    best = final_results[0]

    # Avoid switching away from an established algorithm
    # based on a marginal difference within one tuning run.
    previous = next(
        (r for r in final_results if r["name"] == incumbent),
        None,
    )

    if previous and best["name"] != incumbent:
        gain = (
            previous["latency_ms"] / best["latency_ms"] - 1
        ) * 100

        if gain < args.min_gain_pct:
            best = previous

    legacy = next(
        (r for r in final_results if r["name"] == "legacy"),
        None,
    )


    # The FP16 reference is benchmarked separately in main().
    print(
        f"SELECTED: {best['name']} "
        f"({best['latency_ms'] * 1000:.3f} us)"
    )

    return {
        "batch_size": batch,
        "best": best,
        "baseline": legacy,
        "previous_best": previous,
        "finalists": final_results,
        "passing": screened,
        "failed": failed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="OpenQuant fast two-stage GPU autotuner"
    )
    parser.add_argument(
        "--model",
        default="models/minilm/model.safetensors",
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
    )
    parser.add_argument(
        "--batches",
        default="1,4,16,128",
    )
    parser.add_argument("--screen-ms", type=int, default=3)
    parser.add_argument("--final-ms", type=int, default=12)
    parser.add_argument("--final-repeats", type=int, default=3)
    parser.add_argument("--min-gain-pct", type=float, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="reports/tuning_fast_v06.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    if (
        args.screen_ms < 1
        or args.final_ms < 1
        or args.final_repeats < 1
        or not 0 <= args.min_gain_pct < 100
    ):
        parser.error("Invalid benchmark settings")

    batches = list(dict.fromkeys(
        int(item.strip())
        for item in args.batches.split(",")
    ))

    if not batches or any(batch < 1 for batch in batches):
        parser.error("Batch sizes must be positive")

    if args.group < 2 or args.group % 2:
        parser.error("Group size must be even and >= 2")

    model = Path(args.model)
    if not model.is_file():
        raise FileNotFoundError(model)

    _, layers = load_layers(model)

    if not 0 <= args.layer_index < len(layers):
        parser.error("Layer index out of range")

    layer_name, matrix = layers[args.layer_index]
    weights = np.asarray(matrix, dtype=np.float32)
    n, k = weights.shape

    quantized = quantize_int4_groupwise(
        weights, args.group
    )

    packed = torch.from_numpy(quantized.data).cuda()
    scales = torch.from_numpy(quantized.scales).cuda()

    reconstructed = torch.from_numpy(
        dequantize_int4_groupwise(quantized)
    ).cuda().half()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    report = {
        "project": "OpenQuant",
        "version": "0.6-fast-tuning",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "layer": layer_name,
        "weight_shape": [n, k],
        "group_size": args.group,
        "settings": {
            "screen_ms": args.screen_ms,
            "final_ms": args.final_ms,
            "final_repeats": args.final_repeats,
            "min_gain_pct": args.min_gain_pct,
            "seed": args.seed,
        },
        "results": [],
    }

    started = time.perf_counter()

    print("OPENQUANT FAST AUTOTUNER")
    print("GPU:", report["gpu"])
    print("Layer:", layer_name)
    print("Shape:", (n, k))
    print("Mode:", args.mode)

    for batch in batches:
        x = torch.randn(
            (batch, k),
            device="cuda",
            dtype=torch.float16,
        )

        reference = (x @ reconstructed.T).float()

        entry = tune_batch(
            batch,
            x,
            packed,
            scales,
            n,
            args.group,
            reference,
            args,
        )

        fp16_fn = lambda: x @ reconstructed.T

        entry["fp16_latency_ms"] = median([
            benchmark(fp16_fn, args.final_ms)
            for _ in range(args.final_repeats)
        ])

        report["results"].append(entry)

    report["elapsed_seconds"] = (
        time.perf_counter() - started
    )

    destination = Path(args.output)
    destination.parent.mkdir(
        parents=True, exist_ok=True
    )
    destination.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport:", destination.resolve())
    print(
        "Elapsed:",
        round(report["elapsed_seconds"], 2),
        "seconds",
    )


if __name__ == "__main__":
    main()