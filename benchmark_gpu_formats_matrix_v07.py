
"""Shared-context FP32, FP16, INT8 and INT4 benchmark."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from openquant.quantization import quantize
from openquant.gpu_benchmark import decode_gpu, measure

MIB = 1024 ** 2

ERROR_LIMITS = {
    "FP32": 0.001,
    "FP16": 0.5,
    "INT8": 2.0,
    "INT4": 20.0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1536)
    parser.add_argument("--cols", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    if min(args.rows, args.cols, args.repeats) < 1:
        parser.error("All dimensions and repeats must be positive")

    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    began = time.perf_counter()

    weights_cpu = rng.normal(
        size=(args.rows, args.cols)
    ).astype(np.float32)

    weights = torch.from_numpy(weights_cpu).cuda()
    weights_half = weights.half()

    packed = {}
    for bits in (8, 4):
        q = quantize(weights_cpu, bits)
        packed[bits] = {
            "data": torch.from_numpy(q.data).cuda(),
            "scales": torch.from_numpy(q.scales).cuda(),
            "storage": q.data.nbytes + q.scales.nbytes,
        }

    print("GPU:", torch.cuda.get_device_name(0))
    print("Matrix:", weights_cpu.shape)
    print("Tested formats: FP32, FP16, INT8, INT4")
    print("Repetitions:", args.repeats)

    report_rows = []

    for batch in (1, 4, 128):
        inputs_cpu = rng.normal(
            size=(batch, args.cols)
        ).astype(np.float32)

        x = torch.from_numpy(inputs_cpu).cuda()
        x_half = x.half()

        reference = x @ weights.T

        functions = {
            "FP32": lambda: x @ weights.T,
            "FP16": lambda: x_half @ weights_half.T,
        }

        for bits in (8, 4):
            data = packed[bits]["data"]
            scales = packed[bits]["scales"]

            def execute(qdata=data, qscales=scales, precision=bits):
                decoded = decode_gpu(
                    qdata, qscales, precision,
                    weights_cpu.shape,
                )
                return x @ decoded.T

            functions[f"INT{bits}"] = execute

        print(f"\nBATCH {batch}")
        print("-" * 66)
        print(
            f"{'Format':<10}"
            f"{'Latency ms':>14}"
            f"{'Error %':>14}"
            f"{'Storage MiB':>15}"
        )

        reference_norm = torch.linalg.vector_norm(reference)

        for name, function in functions.items():
            with torch.inference_mode():
                actual = function()
                torch.cuda.synchronize()

                error = (
                    torch.linalg.vector_norm(
                        actual.float() - reference
                    )
                    / reference_norm.clamp_min(1e-12)
                    * 100
                ).item()

                if not math.isfinite(error):
                    raise RuntimeError(
                        f"{name}: non-finite error"
                    )

                if error > ERROR_LIMITS[name]:
                    raise RuntimeError(
                        f"{name}: {error:.3f}% exceeds "
                        f"{ERROR_LIMITS[name]:.3f}%"
                    )

                latency = measure(
                    function, args.repeats
                )

            storage = (
                weights_cpu.nbytes
                if name == "FP32"
                else weights_half.numel() * 2
                if name == "FP16"
                else packed[int(name[3:])]["storage"]
            )

            report_rows.append({
                "batch": batch,
                "format": name,
                "median_ms": float(latency),
                "relative_error_percent": error,
                "storage_bytes": storage,
                "correctness": "PASS",
            })

            print(
                f"{name:<10}"
                f"{latency:>14.5f}"
                f"{error:>14.4f}"
                f"{storage / MIB:>15.3f}"
            )

            del actual

        del x, x_half, reference

    elapsed = time.perf_counter() - began

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "matrix": [args.rows, args.cols],
        "batches": [1, 4, 128],
        "formats": ["FP32", "FP16", "INT8", "INT4"],
        "repeats": args.repeats,
        "wall_seconds": elapsed,
        "results": report_rows,
        "note": (
            "INT8/INT4 include decoding on every call; "
            "this is not fused kernel performance."
        ),
    }

    output = Path("reports/gpu_formats_matrix_v07.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    assert len(report_rows) == 12

    print("\n12 format/batch measurements: PASS")
    print(f"Total wall time: {elapsed:.3f} seconds")
    print("Report:", output.resolve())


if __name__ == "__main__":
    main()
