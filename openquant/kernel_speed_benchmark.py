import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import triton.testing

from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)
from openquant.triton_int4 import int4_linear


def measure(fn):
    try:
        return triton.testing.do_bench_cudagraph(
            fn, rep=30, return_mode="median"
        )
    except Exception as exc:
        print("CUDA Graph benchmark unavailable:", str(exc)[:100])
        return triton.testing.do_bench(
            fn, warmup=30, rep=100, return_mode="median"
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    model = Path("models/minilm/model.safetensors")

    if model.is_file():
        from openquant.model_eval import load_layers
        _, layers = load_layers(model)
        name, matrix = layers[0]
        weights = np.asarray(matrix, dtype=np.float32)
    else:
        name = "synthetic_2048x2048"
        rng = np.random.default_rng(42)
        weights = rng.normal(
            size=(2048, 2048)
        ).astype(np.float32)

    n, k = weights.shape
    group = 32
    q = quantize_int4_groupwise(weights, group)

    packed = torch.from_numpy(q.data).cuda()
    scales = torch.from_numpy(q.scales).cuda()

    restored = torch.from_numpy(
        dequantize_int4_groupwise(q)
    ).cuda().half()

    print("GPU:", torch.cuda.get_device_name(0))
    print("Weights:", name, (n, k))
    print("INT4 group:", group)

    results = []

    for m in (1, 4, 16, 128):
        x = torch.randn(
            (m, k), device="cuda", dtype=torch.float16
        )
        reference = x @ restored.T

        cases = {
            "legacy": lambda: int4_linear(
                x, packed, scales, n, group,
                algorithm="legacy",
            ),
            "packed": lambda: int4_linear(
                x, packed, scales, n, group,
                algorithm="packed",
            ),
        }

        if m <= 4 and k <= 8192:
            cases["gemv"] = lambda: int4_linear(
                x, packed, scales, n, group,
                algorithm="gemv",
            )

        print("\nM =", m, " N =", n, " K =", k)
        print("-" * 60)

        timings = {}

        for mode, fn in cases.items():
            try:
                output = fn()
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    output,
                    reference.float(),
                    rtol=0.02,
                    atol=0.02,
                )

                latency = measure(fn)
                timings[mode] = round(float(latency), 6)

                print(f"{mode:>12}: {latency:.5f} ms  PASS")

            except Exception as exc:
                print(f"{mode:>12}: FAILED {str(exc)[:250]}")

        fp16_ms = measure(lambda: x @ restored.T)
        timings["fp16"] = round(float(fp16_ms), 6)

        print(f"{'fp16':>12}: {fp16_ms:.5f} ms")

        if "legacy" in timings:
            baseline = timings["legacy"]

            for mode in ("gemv", "packed"):
                if mode in timings:
                    speedup = baseline / timings[mode]
                    print(f"{mode} / legacy speedup: {speedup:.2f}x")

        eligible = {
            mode: time for mode, time in timings.items()
            if mode != "fp16"
        }

        winner = (
            min(eligible, key=eligible.get)
            if eligible else None
        )

        results.append({
            "M": m,
            "N": n,
            "K": k,
            "timings_ms": timings,
            "fastest_int4": winner,
        })

    data = {
        "project": "OpenQuant",
        "benchmark": "INT4 accelerated kernel comparison",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "weight_matrix": name,
        "group_size": group,
        "results": results,
    }

    output = Path("reports/kernel_speed_v06.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("\nSaved:", output.resolve())


if __name__ == "__main__":
    main()
