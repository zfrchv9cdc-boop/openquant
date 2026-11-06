from statistics import median

import numpy as np
import torch

from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)
from openquant.group_benchmark import restore_groupwise_torch
from openquant.triton_int4 import int4_linear


CONFIGS = [
    (16, 32, 32, 4),
    (16, 64, 64, 4),
    (32, 32, 64, 4),
    (32, 64, 64, 4),
    (32, 64, 128, 4),
    (32, 128, 64, 4),
    (64, 64, 64, 4),
    (64, 64, 128, 4),
    (64, 128, 64, 8),
    (64, 128, 128, 8),
]


def time_gpu(fn, repetitions=30):
    with torch.inference_mode():
        for _ in range(10):
            result = fn()
            del result

        torch.cuda.synchronize()
        times = []

        for _ in range(repetitions):
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

    rng = np.random.default_rng(42)

    weights = rng.normal(
        size=(2048, 2048)
    ).astype(np.float32)

    inputs = rng.normal(
        size=(128, 2048)
    ).astype(np.float32)

    x32 = torch.from_numpy(inputs).cuda()
    x16 = x32.half()

    w16 = torch.from_numpy(weights).cuda().half()

    q = quantize_int4_groupwise(weights, 32)

    packed = torch.from_numpy(q.data).cuda()
    scales = torch.from_numpy(q.scales).cuda()

    restored16 = torch.from_numpy(
        dequantize_int4_groupwise(q)
    ).cuda().half()

    with torch.inference_mode():
        reference = x16 @ restored16.T

    def old_method():
        restored = restore_groupwise_torch(
            packed, scales, weights.shape, 32
        )
        return x32 @ restored.T

    fp16_ms = time_gpu(lambda: x16 @ w16.T)
    old_ms = time_gpu(old_method)

    print("\nOPENQUANT v0.4.1 - KERNEL TUNING")
    print("GPU:", torch.cuda.get_device_name(0))
    print("FP16 baseline: %.4f ms" % fp16_ms)
    print("Old INT4:      %.4f ms" % old_ms)

    print("\nTesting block configurations")
    print("-" * 65)
    print(
        f"{'BM':>5} {'BN':>5} {'BK':>5} "
        f"{'Warps':>7} {'Latency ms':>13} {'Status':>10}"
    )

    successful = []
    failures = []

    for bm, bn, bk, warps in CONFIGS:

        def run():
            return int4_linear(
                x16,
                packed,
                scales,
                weights.shape[0],
                32,
                block_m=bm,
                block_n=bn,
                block_k=bk,
                num_warps=warps,
            )

        try:
            with torch.inference_mode():
                result = run()
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    result,
                    reference.float(),
                    rtol=0.02,
                    atol=0.02,
                )

                del result

            elapsed = time_gpu(run)

            successful.append(
                (elapsed, bm, bn, bk, warps)
            )

            print(
                f"{bm:>5} {bn:>5} {bk:>5} "
                f"{warps:>7} {elapsed:>13.4f} "
                f"{'PASS':>10}"
            )

        except Exception as exc:
            failures.append(
                (bm, bn, bk, warps, str(exc))
            )
            print(
                f"{bm:>5} {bn:>5} {bk:>5} "
                f"{warps:>7} {'-':>13} {'FAILED':>10}"
            )

    print("-" * 65)

    if successful:
        successful.sort()

        best = successful[0]
        baseline = next(
            (
                item for item in successful
                if item[1:] == (16, 32, 32, 4)
            ),
            None,
        )

        print("\nFASTEST VERIFIED CONFIGURATION")
        print(
            "BM=%d BN=%d BK=%d warps=%d"
            % best[1:]
        )
        print("Latency: %.4f ms" % best[0])

        if baseline:
            print(
                "Speedup over original kernel: %.2fx"
                % (baseline[0] / best[0])
            )

        print(
            "Compared with FP16: %.2fx latency"
            % (best[0] / fp16_ms)
        )

    if failures:
        print("\nFAILED CONFIGURATIONS")
        for bm, bn, bk, warps, reason in failures:
            print(
                f"({bm}, {bn}, {bk}, {warps}): "
                f"{reason[:300]}"
            )


if __name__ == "__main__":
    main()
