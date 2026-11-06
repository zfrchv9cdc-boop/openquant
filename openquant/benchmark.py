
import numpy as np

from openquant.quantization import quantize, dequantize


def benchmark(rows=1024, columns=1024):
    rng = np.random.default_rng(42)

    weights = rng.normal(
        0, 0.2, (rows, columns)
    ).astype(np.float32)

    original_bytes = weights.nbytes

    print(f"Original FP32: {original_bytes:,} bytes")

    for bits in (8, 4):
        result = quantize(weights, bits)
        restored = dequantize(result)

        stored_bytes = (
            result.data.nbytes + result.scales.nbytes
        )

        error = (
            np.linalg.norm(weights - restored)
            / np.linalg.norm(weights)
        )

        savings = 100 * (
            1 - stored_bytes / original_bytes
        )

        print(f"\nINT{bits}")
        print(f"Storage: {stored_bytes:,} bytes")
        print(f"Memory saved: {savings:.2f}%")
        print(f"Relative weight error: {error:.4%}")


if __name__ == "__main__":
    benchmark()
