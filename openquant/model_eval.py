import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from openquant.quantization import quantize, dequantize
from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)


def load_layers(path, contains="", include_embeddings=False):
    path = Path(path)

    if path.suffix.lower() != ".safetensors":
        raise ValueError("Expected a .safetensors checkpoint")

    if not path.is_file():
        raise FileNotFoundError(path)

    tensors = load_file(str(path))
    selected = []

    for name, array in tensors.items():
        if array.ndim != 2 or array.size == 0:
            continue

        if not np.issubdtype(array.dtype, np.floating):
            continue

        if contains and contains.lower() not in name.lower():
            continue

        if not include_embeddings and "embedding" in name.lower():
            continue

        selected.append((name, array))

    selected.sort(key=lambda item: (-item[1].size, item[0]))
    return tensors, selected


def inspect_model(path, limit=12, contains=""):
    if limit < 1:
        raise ValueError("Limit must be positive")

    tensors, selected = load_layers(path, contains=contains)

    return {
        "total_tensors": len(tensors),
        "eligible_matrices": len(selected),
        "layers": [
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "parameters": int(array.size),
            }
            for name, array in selected[:limit]
        ],
    }


def relative_error_pct(reference, actual):
    difference = (
        reference.astype(np.float64)
        - actual.astype(np.float64)
    )

    denominator = np.linalg.norm(
        reference.astype(np.float64)
    )

    if denominator == 0:
        return 0.0 if np.count_nonzero(difference) == 0 else 100.0

    return float(
        100.0 * np.linalg.norm(difference) / denominator
    )


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def analyze_model(
    path,
    max_layers=6,
    batch_size=8,
    group_size=32,
    seed=42,
    contains="",
):
    if max_layers < 1 or batch_size < 1:
        raise ValueError("Limits and batch size must be positive")

    if group_size < 2 or group_size % 2:
        raise ValueError("Group size must be even and >= 2")

    path = Path(path)
    tensors, selected = load_layers(path, contains=contains)

    if not selected:
        raise ValueError("No eligible 2D weight matrices found")

    layers = []

    for index, (name, original) in enumerate(selected[:max_layers]):
        weights = np.ascontiguousarray(
            original, dtype=np.float32
        )

        if not np.isfinite(weights).all():
            raise ValueError(f"Non-finite weights in {name}")

        rows, cols = weights.shape
        rng = np.random.default_rng(seed + index)

        activations = rng.standard_normal(
            (batch_size, cols)
        ).astype(np.float32)

        reference_output = activations @ weights.T

        baseline_bytes = weights.nbytes
        methods = {}

        def record(method, restored, stored_bytes):
            output = activations @ restored.T

            methods[method] = {
                "storage_bytes": int(stored_bytes),
                "compression_ratio": round(
                    baseline_bytes / stored_bytes, 5
                ),
                "memory_reduction_pct": round(
                    100 * (1 - stored_bytes / baseline_bytes), 5
                ),
                "weight_error_pct": round(
                    relative_error_pct(weights, restored), 6
                ),
                "output_error_pct": round(
                    relative_error_pct(reference_output, output), 6
                ),
            }

        record("FP32", weights, baseline_bytes)

        for bits in (8, 4):
            quantized = quantize(weights, bits)
            restored = dequantize(quantized)

            record(
                f"INT{bits}",
                restored,
                quantized.data.nbytes + quantized.scales.nbytes,
            )

        quantized = quantize_int4_groupwise(
            weights, group_size
        )
        restored = dequantize_int4_groupwise(quantized)

        record(
            f"INT4_GROUP_{group_size}",
            restored,
            quantized.data.nbytes + quantized.scales.nbytes,
        )

        layers.append({
            "name": name,
            "shape": [rows, cols],
            "source_dtype": str(original.dtype),
            "parameters": int(weights.size),
            "methods": methods,
        })

    return {
        "schema_version": 1,
        "project": "OpenQuant",
        "version": "0.6.0.dev0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "filename": path.name,
            "file_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "experiment": {
            "seed": seed,
            "batch_size": batch_size,
            "group_size": group_size,
            "input_distribution": "standard_normal",
            "reference": "FP32 matrix multiplication",
            "scope": "Real weights with synthetic activations",
        },
        "total_checkpoint_tensors": len(tensors),
        "eligible_matrices": len(selected),
        "analyzed_matrices": len(layers),
        "layers": layers,
    }
