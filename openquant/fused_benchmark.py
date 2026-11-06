from statistics import median

import numpy as np
import torch

from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)
from openquant.group_benchmark import restore_groupwise_torch
from openquant.triton_int4 import int4_linear


def measure(fn, repeats=40):
    with torch.inference_mode():
        for _ in range(10):
            fn()

        torch.cuda.synchronize()
        samples = []

        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output = fn()
            end.record()

            end.synchronize()
            samples.append(start.elapsed_time(end))
            del output

    return median(samples)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    torch.set_float32_matmul_precision("highest")

    rng = np.random.default_rng(42)

    weights = rng.normal(
        size=(2048, 2048)
    ).astype(np.float32)

    inputs = rng.normal(
        size=(128, 2048)
    ).astype(np.float32)

    x = torch.from_numpy(inputs).cuda()
    w = torch.from_numpy(weights).cuda()

    x16 = x.half()
    w16 = w.half()

    group_size = 32

    q = quantize_int4_groupwise(weights, group_size)

    packed = torch.from_numpy(q.data).cuda()
    scales = torch.from_numpy(q.scales).cuda()

    def fused():
        return int4_linear(
            x16, packed, scales,
            weights.shape[0], group_size
        )

    def old_method():
        restored = restore_groupwise_torch(
            packed, scales, weights.shape, group_size
        )
        return x @ restored.T

    cases = [
        ("FP32", lambda: x @ w.T),
        ("FP16", lambda: x16 @ w16.T),
        ("Old INT4 decode", old_method),
        ("Fused Triton INT4", fused),
    ]

    with torch.inference_mode():
        reference = x @ w.T
        torch.cuda.synchronize()

        print("\nOPENQUANT v0.4 - FUSED GPU BENCHMARK")
        print("GPU:", torch.cuda.get_device_name())
        print("Shape: (128, 2048) x (2048, 2048)")
        print("INT4 group size:", group_size)
        print("-" * 58)

        for name, fn in cases:
            result = fn()
            torch.cuda.synchronize()

            error = (
                torch.linalg.vector_norm(
                    result.float() - reference
                ) / torch.linalg.vector_norm(reference)
                * 100
            ).item()

            del result
            latency = measure(fn)

            print(
                f"{name:<22}"
                f"{latency:>9.4f} ms"
                f"{error:>12.3f}% error"
            )

    storage = q.data.nbytes + q.scales.nbytes

    print("-" * 58)
    print(f"INT4 persistent storage: {storage / 1048576:.3f} MiB")
    print("FP32 persistent storage: 16.000 MiB")
    print("Timing excludes CPU quantization and transfers.")


if __name__ == "__main__":
    main()
