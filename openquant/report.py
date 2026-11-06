import platform
from datetime import datetime, timezone

import numpy as np

from openquant.quantization import quantize, dequantize
from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)


def make_cpu_report(rows=1024, cols=1024,
                    seed=42, group_size=32):

    if rows < 1 or cols < 1:
        raise ValueError("Dimensions must be positive")

    if group_size < 2 or group_size % 2:
        raise ValueError(
            "Group size must be even and >= 2"
        )

    rng = np.random.default_rng(seed)

    weights = rng.normal(
        0, 0.2, (rows, cols)
    ).astype(np.float32)

    baseline_bytes = weights.nbytes
    formats = {}

    def add_result(name, restored, stored_bytes):
        error = (
            np.linalg.norm(weights - restored)
            / np.linalg.norm(weights)
        )

        formats[name] = {
            "storage_bytes": int(stored_bytes),
            "compression_ratio": round(
                baseline_bytes / stored_bytes, 4
            ),
            "memory_reduction_pct": round(
                100 * (1 - stored_bytes / baseline_bytes), 4
            ),
            "relative_weight_error_pct": round(
                100 * float(error), 6
            ),
        }

    add_result("FP32", weights, baseline_bytes)

    for bits in (8, 4):
        q = quantize(weights, bits)
        restored = dequantize(q)

        add_result(
            f"INT{bits}",
            restored,
            q.data.nbytes + q.scales.nbytes,
        )

    q = quantize_int4_groupwise(weights, group_size)
    restored = dequantize_int4_groupwise(q)

    add_result(
        f"INT4_GROUP_{group_size}",
        restored,
        q.data.nbytes + q.scales.nbytes,
    )

    return {
        "schema_version": 1,
        "project": "OpenQuant",
        "version": "0.5.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "device": "cpu",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": seed,
        "weights": {
            "shape": [rows, cols],
            "dtype": "float32",
            "distribution": "normal(0, 0.2)",
        },
        "formats": formats,
    }
