
"""Verify OpenQuant tuned dispatch against its default runtime."""

import json
from pathlib import Path
from statistics import median

import numpy as np
import torch
import triton.testing

from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)
from openquant.model_eval import load_layers
from openquant.triton_int4 import int4_linear
from openquant.tuned_runtime import TunedInt4Linear


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/minilm/model.safetensors"
PROFILE = ROOT / "reports/tuning_fast_v06.json"
OUTPUT = ROOT / "reports/runtime_dispatch_verified.json"


def measure(fn):
    # Warm up the compiled implementation before measuring.
    fn()
    torch.cuda.synchronize()

    samples = [
        float(
            triton.testing.do_bench_cudagraph(
                fn,
                rep=15,
                return_mode="median",
            )
        )
        for _ in range(3)
    ]

    return median(samples)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    if not PROFILE.is_file():
        raise FileNotFoundError(PROFILE)

    _, layers = load_layers(MODEL)
    name, weights = layers[0]

    weights = np.asarray(weights, dtype=np.float32)
    n, k = weights.shape
    group_size = 32

    q = quantize_int4_groupwise(weights, group_size)

    packed = torch.from_numpy(q.data).cuda()
    scales = torch.from_numpy(q.scales).cuda()

    reconstructed = torch.from_numpy(
        dequantize_int4_groupwise(q)
    ).to(device="cuda", dtype=torch.float16)

    tuned = TunedInt4Linear(
        packed,
        scales,
        n,
        k,
        group_size,
        PROFILE,
    )

    torch.manual_seed(42)

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "model_layer": name,
        "shape": [n, k],
        "profile": str(PROFILE),
        "results": [],
    }

    print("OPENQUANT TUNED RUNTIME VERIFICATION")
    print("GPU:", report["gpu"])
    print("Matrix:", name, (n, k))
    print("-" * 70)

    for m in (1, 4, 16, 128):
        x = torch.randn(
            (m, k),
            device="cuda",
            dtype=torch.float16,
        )

        reference = (x @ reconstructed.T).float()

        def default_fn():
            return int4_linear(
                x,
                packed,
                scales,
                n,
                group_size,
                algorithm="auto",
            )

        def tuned_fn():
            return tuned(x)

        default_output = default_fn()
        tuned_output = tuned_fn()

        # Both runtime paths must match the dequantized
        # FP16 reference within the existing test tolerance.
        torch.testing.assert_close(
            default_output,
            reference,
            rtol=0.02,
            atol=0.02,
        )

        torch.testing.assert_close(
            tuned_output,
            reference,
            rtol=0.02,
            atol=0.02,
        )

        default_ms = measure(default_fn)
        tuned_ms = measure(tuned_fn)
        speedup = default_ms / tuned_ms

        config = tuned.configs.get(m, {"algorithm": "auto"})

        result = {
            "batch": m,
            "default_us": round(default_ms * 1000, 4),
            "tuned_us": round(tuned_ms * 1000, 4),
            "speedup": round(speedup, 4),
            "config": config,
            "correctness": "PASS",
        }

        report["results"].append(result)

        print(
            f"M={m:<3} "
            f"default={default_ms * 1000:>8.3f} us  "
            f"tuned={tuned_ms * 1000:>8.3f} us  "
            f"speedup={speedup:.2f}x  PASS"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReport:", OUTPUT)


if __name__ == "__main__":
    main()
