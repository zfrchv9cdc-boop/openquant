
"""OpenQuant v0.7: packed INT4 BERT inference.

Supports groupwise INT4 Linear and Embedding modules.
Compressed matrices remain packed on the GPU.

Requires torch, triton, transformers and safetensors.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from safetensors.numpy import load_file
from transformers import AutoConfig, AutoModel

from openquant.triton_int4 import int4_linear


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
FORMAT = "openquant-int4-checkpoint-v1"


def as_tensor(array):
    """Copy a NumPy array into writable PyTorch memory."""
    return torch.from_numpy(
        np.ascontiguousarray(array).copy()
    )


class PackedInt4Linear(nn.Module):
    """Linear layer that never stores a dequantized weight matrix."""

    def __init__(
        self, original, data, scales, shape, group_size
    ):
        super().__init__()

        n, k = map(int, shape)

        if not isinstance(original, nn.Linear):
            raise TypeError("Expected nn.Linear")

        if (n, k) != (
            original.out_features,
            original.in_features,
        ):
            raise ValueError("Linear weight shape mismatch")

        kp = ((k + group_size - 1) // group_size) * group_size

        if data.size != n * kp // 2:
            raise ValueError("Incorrect packed Linear size")

        if scales.shape != (n, kp // group_size):
            raise ValueError("Incorrect Linear scale shape")

        self.in_features = k
        self.out_features = n
        self.group_size = group_size

        self.register_buffer(
            "packed", as_tensor(data).reshape(-1)
        )

        self.register_buffer(
            "scales", as_tensor(scales)
        )

        bias = (
            None
            if original.bias is None
            else original.bias.detach().clone()
        )

        self.register_buffer("bias", bias)

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError("Linear input dimension mismatch")

        shape = x.shape[:-1]

        # The existing Triton kernel accepts contiguous FP16 inputs.
        flat = x.reshape(-1, self.in_features)
        flat = flat.to(torch.float16).contiguous()

        m = flat.shape[0]

        algorithm = (
            "gemv"
            if m <= 4 and self.in_features <= 8192
            else "packed"
        )

        output = int4_linear(
            flat,
            self.packed,
            self.scales,
            self.out_features,
            self.group_size,
            algorithm=algorithm,
        )

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        return output.reshape(
            *shape, self.out_features
        ).to(x.dtype)


class PackedInt4Embedding(nn.Module):
    """Decode requested embedding rows without unpacking the table."""

    def __init__(
        self, original, data, scales, shape, group_size
    ):
        super().__init__()

        n, k = map(int, shape)

        if not isinstance(original, nn.Embedding):
            raise TypeError("Expected nn.Embedding")

        if (n, k) != (
            original.num_embeddings,
            original.embedding_dim,
        ):
            raise ValueError("Embedding shape mismatch")

        if original.max_norm is not None:
            raise ValueError(
                "Renormalizing embeddings is unsupported"
            )

        kp = ((k + group_size - 1) // group_size) * group_size

        if data.size != n * kp // 2:
            raise ValueError("Incorrect packed Embedding size")

        if scales.shape != (n, kp // group_size):
            raise ValueError("Incorrect Embedding scales")

        self.num_embeddings = n
        self.embedding_dim = k
        self.padded_dim = kp
        self.group_size = group_size
        self.padding_idx = original.padding_idx

        self.register_buffer(
            "packed",
            as_tensor(data).reshape(n, kp // 2),
        )

        self.register_buffer(
            "scales",
            as_tensor(scales),
        )

    def forward(self, input_ids):
        original_shape = input_ids.shape
        ids = input_ids.reshape(-1).long()

        # Only the requested embedding rows are read.
        selected = self.packed.index_select(0, ids)
        selected_scales = self.scales.index_select(0, ids)

        low = (selected & 15).to(torch.int16)
        high = ((selected >> 4) & 15).to(torch.int16)

        values = torch.stack(
            (low, high), dim=-1
        ).reshape(ids.numel(), self.padded_dim)

        signed = torch.where(
            values >= 8, values - 16, values
        ).to(torch.float32)

        expanded_scales = (
            selected_scales.repeat_interleave(
                self.group_size, dim=-1
            )
        )

        decoded = signed * expanded_scales
        decoded = decoded[:, :self.embedding_dim]

        return decoded.reshape(
            *original_shape, self.embedding_dim
        )


def create_backbone():
    """Create the same BERT architecture for both comparisons."""
    config = AutoConfig.from_pretrained(MODEL_ID)

    return AutoModel.from_config(
        config,
        attn_implementation="eager",
    )


def filtered_checkpoint_state(arrays, expected):
    """Account for legacy nonpersistent BERT position buffers."""
    state = {}

    for name, array in arrays.items():
        if name in expected:
            state[name] = as_tensor(array)
        elif not name.endswith(
            (".position_ids", ".token_type_ids")
        ):
            raise ValueError(
                f"Unexpected checkpoint tensor: {name}"
            )

    return state


def load_fp32_reference(source):
    """Load the original checkpoint as an FP32 reference."""
    model = create_backbone()

    arrays = load_file(str(source))
    expected = set(model.state_dict())

    state = filtered_checkpoint_state(arrays, expected)

    result = model.load_state_dict(state, strict=False)

    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "Original model state mismatch: "
            f"{result}"
        )

    return model.eval()


def load_packed_model(archive):
    """Load OpenQuant weights without full-matrix dequantization."""
    archive = Path(archive)
    manifest_path = archive.with_suffix(".json")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )

    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported OpenQuant checkpoint")

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Invalid checkpoint manifest")

    encoded = load_file(str(archive))

    model = create_backbone()
    expected = set(model.state_dict())

    raw = {}
    compressed = []
    seen = set()

    for entry in entries:
        name = entry["name"]
        key = entry["key"]
        mode = entry["mode"]

        if name in seen:
            raise ValueError(f"Duplicate tensor: {name}")

        seen.add(name)

        if mode == "raw":
            value = encoded[key + ".raw"]

            if name in expected:
                raw[name] = as_tensor(value)
            elif not name.endswith(
                (".position_ids", ".token_type_ids")
            ):
                raise ValueError(f"Unexpected raw tensor: {name}")

        elif mode == "int4_groupwise":
            if name not in expected:
                raise ValueError(
                    f"Unknown compressed tensor: {name}"
                )

            compressed.append(entry)

        else:
            raise ValueError(f"Unknown encoding: {mode}")

    result = model.load_state_dict(raw, strict=False)

    quantized_names = {
        entry["name"] for entry in compressed
    }

    unexpected_missing = (
        set(result.missing_keys) - quantized_names
    )

    if unexpected_missing or result.unexpected_keys:
        raise ValueError(
            "Incomplete compressed model: "
            f"{unexpected_missing}, "
            f"{result.unexpected_keys}"
        )

    if set(expected) != set(raw) | quantized_names:
        raise ValueError(
            "Manifest does not cover the full model"
        )

    counts = {"linear": 0, "embedding": 0}

    for entry in compressed:
        name = entry["name"]

        if not name.endswith(".weight"):
            raise ValueError(
                f"Unsupported compressed tensor: {name}"
            )

        path = name[:-len(".weight")]
        parent_path, attr = path.rsplit(".", 1)

        parent = model.get_submodule(parent_path)
        original = getattr(parent, attr)

        key = entry["key"]
        data = encoded[key + ".data"]
        scales = encoded[key + ".scales"]

        args = (
            original,
            data,
            scales,
            entry["shape"],
            int(entry["group_size"]),
        )

        if isinstance(original, nn.Linear):
            replacement = PackedInt4Linear(*args)
            counts["linear"] += 1

        elif isinstance(original, nn.Embedding):
            replacement = PackedInt4Embedding(*args)
            counts["embedding"] += 1

        else:
            raise TypeError(
                f"Cannot execute packed tensor: {name}"
            )

        setattr(parent, attr, replacement)

    return model.eval(), counts


@torch.inference_mode()
def encode_sentences(model, tokens):
    """Generate normalized, mean-pooled sentence embeddings."""
    output = model(**tokens)

    hidden = output.last_hidden_state
    attention = tokens["attention_mask"]

    mask = attention.unsqueeze(-1).to(hidden.dtype)

    pooled = (hidden * mask).sum(dim=1)
    pooled = pooled / mask.sum(dim=1).clamp(min=1)

    return F.normalize(pooled, p=2, dim=1)
