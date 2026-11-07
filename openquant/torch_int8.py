
"""Architecture-independent, weight-only INT8 PyTorch layers.

- Symmetric INT8 per-output-channel weights
- Fused CUDA execution when available
- CPU reference fallback
- Preserves bias and tensor dimensions
- Inference only; no optimizer/gradient support
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from openquant.quantization import quantize


class PackedInt8Linear(nn.Module):
    """Inference-only INT8 replacement for nn.Linear."""

    def __init__(self, original):
        super().__init__()

        if type(original) is not nn.Linear:
            raise TypeError("Expected a standard nn.Linear")

        if original.weight.is_meta:
            raise ValueError(
                "Materialize model weights before quantization"
            )

        self.in_features = original.in_features
        self.out_features = original.out_features

        device = original.weight.device

        weights = (
            original.weight.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
        )

        compressed = quantize(weights, bits=8)

        packed = torch.from_numpy(
            compressed.data.copy()
        ).to(device)

        scales = torch.from_numpy(
            compressed.scales.copy()
        ).reshape(-1).to(device)

        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)

        bias = (
            original.bias.detach().clone()
            if original.bias is not None
            else None
        )

        self.register_buffer("bias", bias)
        self.eval()

    def forward(self, x):
        if self.training:
            raise RuntimeError(
                "PackedInt8Linear is inference-only; "
                "call model.eval()"
            )

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a tensor")

        if x.ndim < 1:
            raise ValueError(
                "Input must have at least one dimension"
            )

        if x.shape[-1] != self.in_features:
            raise ValueError(
                "Incorrect input feature dimension"
            )

        if x.device != self.packed.device:
            raise ValueError(
                "Input and packed weights must share a device"
            )

        if x.dtype not in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError(
                "Expected FP16, FP32 or BF16 activations"
            )

        output_shape = (
            *x.shape[:-1],
            self.out_features,
        )

        if x.numel() == 0:
            return x.new_empty(output_shape)

        flat = x.reshape(-1, self.in_features)

        if x.is_cuda:
            # Lazy import avoids requiring Triton on CPU.
            from openquant.triton_int8 import int8_linear

            output = int8_linear(
                flat.to(torch.float16).contiguous(),
                self.packed,
                self.scales,
            )
        else:
            # Correctness-focused CPU fallback.
            decoded = (
                self.packed.float()
                * self.scales.float().unsqueeze(1)
            )
            output = F.linear(
                flat.float(),
                decoded,
            )

        if self.bias is not None:
            output = output + self.bias.float()

        return output.reshape(output_shape).to(x.dtype)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            "weight_bits=8"
        )


def convert_linear_layers_int8(
    model,
    *,
    inplace=False,
    min_parameters=1024,
    exclude=(),
):
    """Convert standard nn.Linear layers in an eval-mode model.

    Returns (converted_model, report).
    Default is non-destructive and deep-copies the model.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("Expected torch.nn.Module")

    if model.training:
        raise ValueError(
            "Call model.eval() before INT8 conversion"
        )

    if (
        not isinstance(min_parameters, int)
        or isinstance(min_parameters, bool)
        or min_parameters < 0
    ):
        raise ValueError(
            "min_parameters must be a nonnegative integer"
        )

    if isinstance(exclude, str):
        raise TypeError(
            "exclude must be a sequence of module paths"
        )

    excluded = tuple(exclude)

    if not all(
        isinstance(item, str) and item
        for item in excluded
    ):
        raise ValueError(
            "Invalid exclusion path"
        )

    target = model if inplace else deepcopy(model)

    details = []
    before_bytes = 0
    after_bytes = 0
    replacements = {}

    def excluded_path(path):
        return any(
            path == entry
            or path.startswith(entry + ".")
            for entry in excluded
        )

    def replace(parent, prefix=""):
        nonlocal before_bytes, after_bytes

        # Use _modules so shared aliases can reuse one
        # compressed replacement object.
        for name, child in list(parent._modules.items()):
            if child is None:
                continue

            path = f"{prefix}.{name}" if prefix else name

            if excluded_path(path):
                continue

            if type(child) is nn.Linear:
                if child.weight.numel() < min_parameters:
                    continue

                identity = id(child)

                if identity in replacements:
                    parent._modules[name] = replacements[identity]
                    continue

                compressed = PackedInt8Linear(child)
                replacements[identity] = compressed

                original_bytes = (
                    child.weight.numel()
                    * child.weight.element_size()
                )

                compressed_bytes = (
                    compressed.packed.numel()
                    * compressed.packed.element_size()
                    + compressed.scales.numel()
                    * compressed.scales.element_size()
                )

                before_bytes += original_bytes
                after_bytes += compressed_bytes

                details.append({
                    "name": path,
                    "in_features": child.in_features,
                    "out_features": child.out_features,
                    "original_weight_bytes": original_bytes,
                    "int8_weight_bytes": compressed_bytes,
                })

                parent._modules[name] = compressed

            else:
                replace(child, path)

    if type(target) is nn.Linear:
        # Support a standalone linear layer too.
        if target.weight.numel() >= min_parameters:
            compressed = PackedInt8Linear(target)
            original_bytes = (
                target.weight.numel()
                * target.weight.element_size()
            )
            compressed_bytes = (
                compressed.packed.numel()
                * compressed.packed.element_size()
                + compressed.scales.numel()
                * compressed.scales.element_size()
            )
            before_bytes = original_bytes
            after_bytes = compressed_bytes
            details.append({
                "name": "<root>",
                "in_features": target.in_features,
                "out_features": target.out_features,
                "original_weight_bytes": before_bytes,
                "int8_weight_bytes": after_bytes,
            })
            target = compressed
    else:
        replace(target)

    saved_percent = (
        100 * (1 - after_bytes / before_bytes)
        if before_bytes
        else 0.0
    )

    report = {
        "converted_layers": len(details),
        "original_weight_bytes": before_bytes,
        "int8_weight_bytes": after_bytes,
        "weight_storage_saved_percent": saved_percent,
        "layers": details,
        "note": (
            "Storage covers converted weight tensors only; "
            "biases and unconverted layers are excluded."
        ),
    }

    return target, report
