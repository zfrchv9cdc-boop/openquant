
"""Checkpoint inspection and corruption detection tests."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from torch import nn

from openquant import cli
from openquant.torch_int8 import convert_linear_layers_int8
from openquant.int8_checkpoint_v07 import save_int8_checkpoint
from openquant.checkpoint_info_v07 import inspect_checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    ).eval()

    packed, _ = convert_linear_layers_int8(
        model, min_parameters=0
    )

    path = tmp_path / "model.safetensors"
    save_int8_checkpoint(packed, path)

    return path


def read_manifest(path):
    manifest_path = path.with_suffix(".json")
    return json.loads(
        manifest_path.read_text(encoding="utf-8")
    )


def write_manifest(path, data):
    path.with_suffix(".json").write_text(
        json.dumps(data),
        encoding="utf-8",
    )


def test_valid_checkpoint(checkpoint):
    info = inspect_checkpoint(checkpoint)

    assert info["sha256_verified"] is True
    assert info["metadata_validation"] == "PASS"
    assert info["compressed_module_paths"] == 2
    assert info["unique_compressed_layers"] == 2
    assert info["payload_bytes"] > 0


def test_human_readable_cli(checkpoint, capsys):
    assert cli.main([
        "checkpoint-info",
        "--input", str(checkpoint),
    ]) == 0

    output = capsys.readouterr().out
    assert "SHA-256: PASS" in output
    assert "Compressed module paths: 2" in output


def test_json_cli(checkpoint, capsys):
    assert cli.main([
        "checkpoint-info",
        "--input", str(checkpoint),
        "--json",
    ]) == 0

    output = capsys.readouterr().out
    report = json.loads(output)

    assert report["sha256_verified"] is True
    assert report["compressed_module_paths"] == 2


def test_payload_corruption(checkpoint):
    with checkpoint.open("ab") as stream:
        stream.write(b"modified")

    with pytest.raises(ValueError, match="SHA-256"):
        inspect_checkpoint(checkpoint)


def test_missing_manifest(checkpoint):
    checkpoint.with_suffix(".json").unlink()

    with pytest.raises(FileNotFoundError):
        inspect_checkpoint(checkpoint)


def test_invalid_extension(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"not-a-checkpoint")

    with pytest.raises(ValueError):
        inspect_checkpoint(path)


@pytest.mark.parametrize(
    "case",
    [
        "wrong_format",
        "wrong_filename",
        "wrong_digest",
        "duplicate_paths",
        "invalid_dimensions",
        "invalid_count",
        "duplicate_tensor_keys",
    ],
)
def test_invalid_metadata(checkpoint, case):
    metadata = read_manifest(checkpoint)

    if case == "wrong_format":
        metadata["format"] = "invalid"

    elif case == "wrong_filename":
        metadata["tensor_file"] = "other.safetensors"

    elif case == "wrong_digest":
        metadata["tensor_sha256"] = "0" * 64

    elif case == "duplicate_paths":
        metadata["layers"].append(
            dict(metadata["layers"][0])
        )
        metadata["compressed_paths"] += 1

    elif case == "invalid_dimensions":
        metadata["layers"][0]["in_features"] = -1

    elif case == "invalid_count":
        metadata["compressed_paths"] = 999

    elif case == "duplicate_tensor_keys":
        metadata["tensor_keys"].append(
            metadata["tensor_keys"][0]
        )

    write_manifest(checkpoint, metadata)

    with pytest.raises(ValueError):
        inspect_checkpoint(checkpoint)
