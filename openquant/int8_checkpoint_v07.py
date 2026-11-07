
"""Portable OpenQuant INT8 PyTorch checkpoints.

Checkpoint:
    model.safetensors - tensor data
    model.json        - format, architecture metadata, checksum

Requires a compatible nn.Module architecture to restore.
Never executes model code from checkpoint files.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from openquant.torch_int8 import PackedInt8Linear

FORMAT = "openquant-torch-int8-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(model):
    """Return every path, including shared-module aliases."""
    return list(model.named_modules(remove_duplicate=False))


def _packed_layers(model):
    result = []
    for name, module in _paths(model):
        if isinstance(module, PackedInt8Linear):
            result.append({
                "path": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
            })
    return result


def save_int8_checkpoint(model, output):
    """Save an already compressed model to SafeTensors."""

    if not isinstance(model, nn.Module):
        raise TypeError("Expected nn.Module")

    if model.training:
        raise ValueError("Model must be in eval mode")

    output = Path(output)

    if output.suffix != ".safetensors":
        raise ValueError(
            "Output must end in .safetensors"
        )

    manifest_path = output.with_suffix(".json")

    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            "Checkpoint already exists"
        )

    layers = _packed_layers(model)

    if not layers:
        raise ValueError(
            "No PackedInt8Linear layers found"
        )

    from safetensors.torch import save_file

    tensors = {}

    # Clone each tensor so shared storage does not
    # prevent SafeTensors serialization.
    for name, value in model.state_dict().items():
        if value.is_meta:
            raise ValueError(
                f"Cannot serialize meta tensor: {name}"
            )

        tensors[name] = (
            value.detach()
            .to("cpu")
            .contiguous()
            .clone()
        )

    if not tensors:
        raise ValueError("Model has no persistent state")

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Save compressed tensor payload first.
    save_file(tensors, str(output))

    manifest = {
        "format": FORMAT,
        "tensor_file": output.name,
        "tensor_sha256": sha256_file(output),
        "tensor_keys": sorted(tensors),
        "layers": layers,
        "compressed_paths": len(layers),
        "unique_compressed_layers": len({
            id(module)
            for _, module in _paths(model)
            if isinstance(module, PackedInt8Linear)
        }),
    }

    try:
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
    except Exception:
        # Avoid leaving an unusable orphan checkpoint.
        output.unlink(missing_ok=True)
        raise

    print("INT8 checkpoint saved:", output.resolve())
    print("Manifest:", manifest_path.resolve())

    return manifest


def _new_packed_layer(original):
    """Build compressed buffers without quantizing dummy weights.

    The nn.Linear skeleton supplies architecture only.
    No FP32 model weights are loaded or converted here.
    """
    if type(original) is not nn.Linear:
        raise TypeError(
            "Checkpoint expects a standard nn.Linear"
        )

    module = PackedInt8Linear.__new__(PackedInt8Linear)
    nn.Module.__init__(module)

    module.in_features = original.in_features
    module.out_features = original.out_features

    device = original.weight.device

    module.register_buffer(
        "packed",
        torch.empty(
            (original.out_features, original.in_features),
            dtype=torch.int8,
            device=device,
        ),
    )

    module.register_buffer(
        "scales",
        torch.empty(
            original.out_features,
            dtype=torch.float32,
            device=device,
        ),
    )

    if original.bias is None:
        bias = None
    else:
        bias = torch.empty(
            original.out_features,
            dtype=original.bias.dtype,
            device=device,
        )

    module.register_buffer("bias", bias)
    module.eval()

    return module


def _get_module(model, path):
    return model if path == "" else model.get_submodule(path)


def _set_module(model, path, replacement):
    if path == "":
        return replacement

    parent_name, _, child_name = path.rpartition(".")

    parent = (
        model.get_submodule(parent_name)
        if parent_name
        else model
    )

    if child_name not in parent._modules:
        raise ValueError(
            f"Unknown module path: {path}"
        )

    parent._modules[child_name] = replacement

    return model


def load_int8_checkpoint(
    skeleton,
    source,
    *,
    device=None,
    inplace=False,
):
    """Restore compressed weights into a compatible architecture.

    Returns (model, manifest). Does not need original FP32
    checkpoint weights, but the caller must supply the model
    architecture with corresponding nn.Linear modules.
    """

    if not isinstance(skeleton, nn.Module):
        raise TypeError("Expected nn.Module skeleton")

    if skeleton.training:
        raise ValueError("Call skeleton.eval() first")

    source = Path(source)

    if source.suffix != ".safetensors":
        raise ValueError(
            "Source must end in .safetensors"
        )

    manifest_path = source.with_suffix(".json")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.get("format") != FORMAT:
        raise ValueError(
            "Unsupported checkpoint format"
        )

    if manifest.get("tensor_file") != source.name:
        raise ValueError(
            "Checkpoint filename does not match manifest"
        )

    if sha256_file(source) != manifest.get(
        "tensor_sha256"
    ):
        raise ValueError(
            "Checkpoint SHA-256 verification failed"
        )

    layers = manifest.get("layers")

    if not isinstance(layers, list) or not layers:
        raise ValueError(
            "Manifest contains no compressed layers"
        )

    if not all(
        isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and type(entry.get("in_features")) is int
        and type(entry.get("out_features")) is int
        and type(entry.get("has_bias")) is bool
        for entry in layers
    ):
        raise ValueError(
            "Invalid compressed-layer metadata"
        )

    from safetensors.torch import load_file
    tensors = load_file(str(source), device="cpu")

    if sorted(tensors) != manifest.get("tensor_keys"):
        raise ValueError(
            "Checkpoint tensor keys do not match manifest"
        )

    if len({
        entry["path"] for entry in layers
    }) != len(layers):
        raise ValueError(
            "Duplicate compressed-layer paths"
        )

    # Validate architecture before modifying anything.
    for entry in layers:
        name = entry["path"]

        try:
            module = _get_module(skeleton, name)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown architecture module: {name}"
            ) from exc

        if type(module) is not nn.Linear:
            raise ValueError(
                f"Expected nn.Linear at {name}"
            )

        if (
            module.in_features != entry["in_features"]
            or module.out_features != entry["out_features"]
            or (module.bias is not None)
            != entry["has_bias"]
        ):
            raise ValueError(
                f"Incompatible layer dimensions/bias: {name}"
            )

    result = (
        skeleton if inplace
        else deepcopy(skeleton)
    )

    replacements = {}

    # Replace original Linear modules with packed-buffer
    # modules. Preserve repeated aliases to the same module.
    for entry in layers:
        name = entry["path"]
        original = _get_module(result, name)

        identity = id(original)

        if identity in replacements:
            replacement = replacements[identity]
        else:
            replacement = _new_packed_layer(original)
            replacements[identity] = replacement

        result = _set_module(
            result, name, replacement
        )

    expected = result.state_dict()

    if set(expected) != set(tensors):
        missing = sorted(set(expected) - set(tensors))
        extra = sorted(set(tensors) - set(expected))
        raise ValueError(
            f"State keys mismatch: missing={missing}, "
            f"extra={extra}"
        )

    for name, target in expected.items():
        saved = tensors[name]

        if target.shape != saved.shape:
            raise ValueError(
                f"Tensor shape mismatch: {name}"
            )

        if target.dtype != saved.dtype:
            raise ValueError(
                f"Tensor dtype mismatch: {name}"
            )

    result.load_state_dict(
        tensors,
        strict=True,
    )

    if device is not None:
        result = result.to(device)

    result.eval()

    print("INT8 checkpoint loaded:", source.resolve())
    print("Compressed module paths:", len(layers))

    return result, manifest
