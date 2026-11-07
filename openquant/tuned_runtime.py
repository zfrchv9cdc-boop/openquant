"""Runtime selection from a validated OpenQuant tuning report."""
import json
from pathlib import Path

import torch

from openquant.triton_int4 import int4_linear


class TunedInt4Linear:
    def __init__(
        self,
        packed,
        scales,
        out_features,
        in_features,
        group_size,
        profile_path,
    ):
        report = json.loads(
            Path(profile_path).read_text(encoding="utf-8")
        )

        expected_shape = [out_features, in_features]

        if report["weight_shape"] != expected_shape:
            raise ValueError("Tuning profile weight shape mismatch")

        if report["group_size"] != group_size:
            raise ValueError("Tuning profile group size mismatch")

        device_name = torch.cuda.get_device_name(packed.device)

        if report["gpu"] != device_name:
            raise ValueError("Tuning profile GPU mismatch")

        self.packed = packed
        self.scales = scales
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size

        self.configs = {}

        for entry in report["results"]:
            best = entry.get("best")

            if best is not None:
                self.configs[entry["batch_size"]] = best["config"]

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError("Input dimension mismatch")

        config = self.configs.get(
            x.shape[0], {"algorithm": "auto"}
        )

        return int4_linear(
            x,
            self.packed,
            self.scales,
            self.out_features,
            self.group_size,
            **config,
        )
