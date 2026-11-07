"""Pack and unpack experimental groupwise INT4 Safetensors checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from openquant.groupwise import (
    Groupwise4,
    dequantize_int4_groupwise,
    quantize_int4_groupwise,
)

FORMAT = "openquant-int4-checkpoint-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_checkpoint(source, output, group_size=32, min_elements=4096):
    source = Path(source).resolve()
    output = Path(output).resolve()
    manifest_path = output.with_suffix(".json")

    if source.suffix != ".safetensors" or output.suffix != ".safetensors":
        raise ValueError("Input and output must be .safetensors files")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output or output.exists() or manifest_path.exists():
        raise FileExistsError("Choose a new, unused output filename")
    if group_size < 2 or group_size % 2 or min_elements < 1:
        raise ValueError("Group size must be even >= 2; min_elements >= 1")

    tensors = load_file(str(source))
    encoded = {}
    entries = []
    quantized = 0

    for index, (name, array) in enumerate(sorted(tensors.items())):
        key = f"tensor_{index:05d}"
        eligible = (
            array.ndim == 2
            and array.size >= min_elements
            and array.dtype in (np.float16, np.float32)
        )
        entry = {"name": name, "key": key, "dtype": str(array.dtype)}

        if eligible:
            q = quantize_int4_groupwise(array, group_size)
            encoded[key + ".data"] = np.ascontiguousarray(q.data)
            encoded[key + ".scales"] = np.ascontiguousarray(q.scales)
            entry.update({
                "mode": "int4_groupwise",
                "shape": list(array.shape),
                "group_size": group_size,
            })
            quantized += 1
        else:
            encoded[key + ".raw"] = np.ascontiguousarray(array)
            entry["mode"] = "raw"

        entries.append(entry)

    manifest = {
        "format": FORMAT,
        "source_filename": source.name,
        "source_sha256": file_sha256(source),
        "source_bytes": source.stat().st_size,
        "quantized_count": quantized,
        "total_tensors": len(entries),
        "entries": entries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(encoded, str(output))
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    stored_bytes = output.stat().st_size + manifest_path.stat().st_size
    return {
        "quantized_tensors": quantized,
        "total_tensors": len(entries),
        "original_bytes": source.stat().st_size,
        "packed_bytes_with_manifest": stored_bytes,
        "reduction_pct": round(
            100 * (1 - stored_bytes / source.stat().st_size), 3
        ),
        "packed_file": str(output),
    }


def unpack_checkpoint(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    manifest_path = source.with_suffix(".json")

    if output.suffix != ".safetensors" or source == output or output.exists():
        raise ValueError("Choose a new .safetensors output filename")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("Unrecognized OpenQuant checkpoint format")

    encoded = load_file(str(source))
    restored = {}

    for entry in manifest["entries"]:
        key = entry["key"]

        if entry["mode"] == "raw":
            array = encoded[key + ".raw"]
        elif entry["mode"] == "int4_groupwise":
            q = Groupwise4(
                data=encoded[key + ".data"],
                scales=encoded[key + ".scales"],
                shape=tuple(entry["shape"]),
                group_size=int(entry["group_size"]),
            )
            array = dequantize_int4_groupwise(q).astype(entry["dtype"])
        else:
            raise ValueError("Unknown encoding mode")

        restored[entry["name"]] = np.ascontiguousarray(array)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(restored, str(output))
    return {
        "restored_tensors": len(restored),
        "restored_file": str(output),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OpenQuant checkpoint compression"
    )
    parser.add_argument("action", choices=["pack", "unpack"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--min-elements", type=int, default=4096)
    args = parser.parse_args(argv)

    if args.action == "pack":
        result = pack_checkpoint(
            args.input, args.output, args.group_size, args.min_elements
        )
    else:
        result = unpack_checkpoint(args.input, args.output)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
