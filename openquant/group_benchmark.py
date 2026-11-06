import numpy as np
import torch

from statistics import median

from openquant.quantization import quantize
from openquant.gpu_benchmark import decode_gpu
from openquant.groupwise import quantize_int4_groupwise


def restore_groupwise_torch(data, scales, shape, group_size):
    rows, cols = shape

    low = (data & 15).to(torch.int16)
    high = ((data >> 4) & 15).to(torch.int16)

    raw = torch.stack(
        (low, high), dim=1
    ).reshape(-1)

    signed = torch.where(
        raw >= 8, raw - 16, raw
    ).to(torch.float32)

    full = (
        signed.reshape(rows, -1, group_size)
        * scales.unsqueeze(-1)
    ).reshape(rows, -1)

    return full[:, :cols]


def time_gpu(fn, repeats=30):
    for _ in range(10):
        fn()

    torch.cuda.synchronize()
    times = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = fn()
        end.record()

        end.synchronize()
        times.append(start.elapsed_time(end))
        del result

    return median(times)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

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

    operations = [
        ("FP32", lambda: x @ w.T, w.numel() * 4),
        ("FP16", lambda: x16 @ w16.T, w16.numel() * 2)
    ]

    rowwise = quantize(weights, 4)

    row_data = torch.from_numpy(rowwise.data).cuda()
    row_scales = torch.from_numpy(rowwise.scales).cuda()

    def rowwise_forward():
        restored = decode_gpu(
            row_data, row_scales, 4, weights.shape
        )
        return x @ restored.T

    operations.append((
        "INT4 rowwise",
        rowwise_forward,
        rowwise.data.nbytes + rowwise.scales.nbytes
    ))

    for group_size in (64, 32, 16):
        q = quantize_int4_groupwise(
            weights, group_size
        )

        data = torch.from_numpy(q.data).cuda()
        scales = torch.from_numpy(q.scales).cuda()

        def forward(
            data=data,
            scales=scales,
            group_size=group_size
        ):
            restored = restore_groupwise_torch(
                data, scales, weights.shape, group_size
            )
            return x @ restored.T

        operations.append((
            f"INT4 group {group_size}",
            forward,
            q.data.nbytes + q.scales.nbytes
        ))

    with torch.inference_mode():
        reference = x @ w.T
        torch.cuda.synchronize()

        print("\nOPENQUANT v0.3 — GROUPWISE INT4")
        print("GPU:", torch.cuda.get_device_name(0))
        print("-" * 68)

        print(
            f"{'Method':<20}"
            f"{'Storage MiB':>13}"
            f"{'Latency ms':>13}"
            f"{'Error %':>13}"
        )

        for name, fn, storage in operations:
            output = fn()

            error = (
                torch.linalg.vector_norm(
                    output.float() - reference
                ) / torch.linalg.vector_norm(reference)
                * 100
            ).item()

            del output
            elapsed = time_gpu(fn)

            print(
                f"{name:<20}"
                f"{storage / 1048576:>13.3f}"
                f"{elapsed:>13.3f}"
                f"{error:>13.3f}"
            )

    print("\nQuantized methods decode to FP32 before GEMM.")
    print("Storage excludes temporary buffers.")


if __name__ == "__main__":
    main()
