
"""Validated, shape-specific INT4 runtime for OpenQuant v0.6."""

import json
import math
from pathlib import Path

import torch
import triton

from openquant.triton_int4 import int4_linear


ALGORITHMS = {"legacy", "gemv", "packed", "batched"}
TILE_KEYS = {"block_m", "block_n", "block_k", "num_warps"}


def _power_of_two(value):
    return (
        type(value) is int
        and value > 0
        and (value & (value - 1)) == 0
    )


def _validate_config(config, batch, in_features):
    if not isinstance(config, dict):
        raise ValueError("Kernel configuration must be an object")

    algorithm = config.get("algorithm")

    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown tuned kernel algorithm")

    permitted = {"algorithm"}

    if algorithm in {"legacy", "packed"}:
        permitted |= TILE_KEYS
    elif algorithm == "batched":
        permitted |= {"block_n", "num_warps"}

    unknown = set(config) - permitted
    if unknown:
        raise ValueError(
            f"Unsupported tuning configuration keys: {sorted(unknown)}"
        )

    for name in TILE_KEYS & set(config):
        if not _power_of_two(config[name]):
            raise ValueError(f"Invalid tile parameter: {name}")

    if config.get("num_warps", 4) > 16:
        raise ValueError("Unsupported warp count")

    if algorithm == "packed":
        if config.get("block_k", 64) < 32:
            raise ValueError("Packed block_k must be >= 32")

    if algorithm == "gemv" and in_features > 8192:
        raise ValueError("GEMV does not support this K")

    if algorithm == "batched":
        if not 1 <= batch <= 4:
            raise ValueError("Batched GEMV supports M=1..4")

        if in_features > 2048:
            raise ValueError("Batched GEMV supports K<=2048")

        if config.get("block_n") not in (2, 4, 8, 16, 32):
            raise ValueError("Invalid batched output tile")

    return dict(config)


class TunedInt4Linear:
    """Dispatch validated kernels from a GPU-specific tuning report."""

    def __init__(
        self,
        packed,
        scales,
        out_features,
        in_features,
        group_size,
        profile_path,
    ):
        profile = Path(profile_path).resolve()

        report = json.loads(
            profile.read_text(encoding="utf-8-sig")
        )

        if not isinstance(report, dict):
            raise ValueError("Invalid tuning report")

        expected_shape = [out_features, in_features]

        if report.get("weight_shape") != expected_shape:
            raise ValueError("Tuning profile weight shape mismatch")

        if report.get("group_size") != group_size:
            raise ValueError("Tuning profile group size mismatch")

        gpu = torch.cuda.get_device_name(packed.device)

        if report.get("gpu") != gpu:
            raise ValueError("Tuning profile GPU mismatch")

        if (
            report.get("torch_version") is not None
            and report["torch_version"] != torch.__version__
        ):
            raise ValueError("PyTorch version mismatch")

        if (
            report.get("triton_version") is not None
            and report["triton_version"] != triton.__version__
        ):
            raise ValueError("Triton version mismatch")

        self.packed = packed
        self.scales = scales
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.profile_path = profile
        self.configs = {}

        results = report.get("results")

        if not isinstance(results, list):
            raise ValueError("Tuning report has no results list")

        seen_batches = set()

        for entry in results:
            if not isinstance(entry, dict):
                raise ValueError("Invalid tuning result entry")

            batch = entry.get("batch_size")

            if type(batch) is not int or batch < 1:
                raise ValueError("Invalid tuning batch size")

            if batch in seen_batches:
                raise ValueError("Duplicate tuning batch size")

            seen_batches.add(batch)

            best = entry.get("best")

            if best is None:
                continue

            if not isinstance(best, dict):
                raise ValueError("Invalid winning kernel record")

            latency = best.get("latency_ms")

            if (
                not isinstance(latency, (int, float))
                or not math.isfinite(latency)
                or latency <= 0
            ):
                raise ValueError("Invalid winning kernel latency")

            config = _validate_config(
                best.get("config"),
                batch,
                in_features,
            )

            self.configs[batch] = config

    def selected_config(self, batch_size):
        """Return the selected configuration for a batch."""
        return dict(
            self.configs.get(
                batch_size,
                {"algorithm": "auto"},
            )
        )

    def __call__(self, x):
        if x.ndim != 2:
            raise ValueError("Input must be a matrix")

        if x.shape[1] != self.in_features:
            raise ValueError("Input dimension mismatch")

        config = self.selected_config(x.shape[0])

        return int4_linear(
            x,
            self.packed,
            self.scales,
            self.out_features,
            self.group_size,
            **config,
        )
