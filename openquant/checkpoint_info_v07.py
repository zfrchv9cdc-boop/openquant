
"""Inspect OpenQuant INT8 checkpoints without loading model code."""

import hashlib
import hmac
import json
import re
from pathlib import Path

FORMAT = "openquant-torch-int8-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(4 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(source):
    source = Path(source)

    if source.suffix != ".safetensors":
        raise ValueError("Expected a .safetensors checkpoint")

    if not source.is_file():
        raise FileNotFoundError(source)

    manifest_path = source.with_suffix(".json")

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    metadata = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(metadata, dict):
        raise ValueError("Manifest must be a JSON object")

    if metadata.get("format") != FORMAT:
        raise ValueError("Unsupported checkpoint format")

    if metadata.get("tensor_file") != source.name:
        raise ValueError("Checkpoint filename mismatch")

    expected_hash = metadata.get("tensor_sha256")

    if not (
        isinstance(expected_hash, str)
        and re.fullmatch(r"[a-fA-F0-9]{64}", expected_hash)
    ):
        raise ValueError("Invalid SHA-256 field")

    actual_hash = file_sha256(source)

    if not hmac.compare_digest(
        expected_hash.lower(), actual_hash
    ):
        raise ValueError(
            "SHA-256 mismatch: checkpoint bytes were modified"
        )

    keys = metadata.get("tensor_keys")

    if not (
        isinstance(keys, list)
        and keys
        and all(isinstance(k, str) and k for k in keys)
        and len(set(keys)) == len(keys)
    ):
        raise ValueError("Invalid checkpoint tensor key list")

    layers = metadata.get("layers")

    if not isinstance(layers, list) or not layers:
        raise ValueError("No compressed layers in manifest")

    paths = set()

    for entry in layers:
        if not isinstance(entry, dict):
            raise ValueError("Invalid layer entry")

        path = entry.get("path")
        rows = entry.get("out_features")
        cols = entry.get("in_features")
        bias = entry.get("has_bias")

        if not isinstance(path, str):
            raise ValueError("Invalid layer path")

        if path in paths:
            raise ValueError("Duplicate layer path")

        paths.add(path)

        if (
            type(rows) is not int
            or type(cols) is not int
            or rows < 1
            or cols < 1
            or type(bias) is not bool
        ):
            raise ValueError(
                f"Invalid layer dimensions or bias: {path}"
            )

    unique_count = metadata.get("unique_compressed_layers")

    if (
        type(unique_count) is not int
        or not 1 <= unique_count <= len(layers)
    ):
        raise ValueError("Invalid unique layer count")

    if metadata.get("compressed_paths") != len(layers):
        raise ValueError("Compressed path count mismatch")

    result = {
        "format": FORMAT,
        "checkpoint": str(source.resolve()),
        "payload_bytes": source.stat().st_size,
        "manifest_bytes": manifest_path.stat().st_size,
        "sha256": actual_hash,
        "sha256_verified": True,
        "tensor_keys_declared": len(keys),
        "compressed_module_paths": len(layers),
        "unique_compressed_layers": unique_count,
        "layers": layers,
        "metadata_validation": "PASS",
    }

    return result


def configure_checkpoint_info(commands):
    parser = commands.add_parser(
        "checkpoint-info",
        help="Inspect and verify a portable INT8 checkpoint",
    )
    parser.add_argument(
        "--input", required=True, type=Path
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print structured JSON",
    )


def run_checkpoint_info(args):
    result = inspect_checkpoint(args.input)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("OPENQUANT INT8 CHECKPOINT")
        print("-" * 48)
        print("File:", result["checkpoint"])
        print("Payload:", result["payload_bytes"], "bytes")
        print("Declared tensors:", result["tensor_keys_declared"])
        print(
            "Unique compressed layers:",
            result["unique_compressed_layers"],
        )
        print(
            "Compressed module paths:",
            result["compressed_module_paths"],
        )
        print("SHA-256: PASS")
        print("Manifest structure: PASS")

        for layer in result["layers"]:
            print(
                f"  {layer['path'] or '<root>'}: "
                f"{layer['in_features']} -> "
                f"{layer['out_features']}"
            )

    return 0
