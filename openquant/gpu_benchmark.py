import argparse
from statistics import median

import numpy as np
import torch

from openquant.quantization import quantize


def decode_gpu(data, scales, bits, shape):
    if bits == 8:
        values = data.reshape(shape).float()

    elif bits == 4:
        low = (data & 15).to(torch.int16)
        high = ((data >> 4) & 15).to(torch.int16)

        values = torch.stack(
            (low, high), dim=1
        ).reshape(-1)

        values = values[:shape[0] * shape[1]]
        values = torch.where(
            values >= 8, values - 16, values
        ).reshape(shape).float()

    else:
        raise ValueError("Expected 4 or 8 bits")

    return values * scales


def measure(fn, repeats):
    for _ in range(10):
        fn()

    torch.cuda.synchronize()

    timings = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = fn()
        end.record()

        end.synchronize()
        timings.append(start.elapsed_time(end))
        del result

    return median(timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if min(args.rows, args.cols, args.batch, args.repeats) <= 0:
        parser.error("All dimensions and repeats must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    device = "cuda"
    torch.set_float32_matmul_precision("highest")

    rng = np.random.default_rng(42)

    w_np = rng.normal(
        size=(args.rows, args.cols)
    ).astype(np.float32)

    x_np = rng.normal(
        size=(args.batch, args.cols)
    ).astype(np.float32)

    w = torch.from_numpy(w_np).to(device)
    x = torch.from_numpy(x_np).to(device)

    reference = x @ w.T
    torch.cuda.synchronize()

    tests = [
        ("FP32", lambda: x @ w.T, w.numel() * 4),
        (
            "FP16",
            lambda: x.half() @ w.half().T,
            w.numel() * 2
        )
    ]

    for bits in (8, 4):
        q = quantize(w_np, bits)

        data = torch.from_numpy(q.data).to(device)
        scales = torch.from_numpy(q.scales).to(device)

        stored = q.data.nbytes + q.scales.nbytes

        def run(data=data, scales=scales, bits=bits):
            restored = decode_gpu(
                data, scales, bits, w_np.shape
            )
            return x @ restored.T

        tests.append((f"INT{bits}", run, stored))

    print("\nOPENQUANT GPU BENCHMARK")
    print("=" * 65)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("Matrix:", w_np.shape)
    print()

    print(
        f"{'Format':<12}"
        f"{'Storage MiB':>14}"
        f"{'Median ms':>14}"
        f"{'Error %':>14}"
    )

    with torch.inference_mode():
        for name, fn, stored in tests:
            output = fn()

            error = (
                torch.linalg.vector_norm(
                    output.float() - reference
                )
                / torch.linalg.vector_norm(reference)
                * 100
            ).item()

            del output

            elapsed = measure(fn, args.repeats)

            print(
                f"{name:<12}"
                f"{stored / 1048576:>14.3f}"
                f"{elapsed:>14.3f}"
                f"{error:>14.3f}"
            )

    print("\nINT4/INT8 decode to FP32 before multiplication.")
    print("Storage excludes temporary computation buffers.")


if __name__ == "__main__":
    main()
