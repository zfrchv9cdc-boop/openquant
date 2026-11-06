import numpy as np
import pytest
from safetensors.numpy import save_file

from openquant.model_eval import (
    inspect_model,
    analyze_model,
    relative_error_pct,
)


@pytest.fixture
def checkpoint(tmp_path):
    rng = np.random.default_rng(42)

    tensors = {
        "encoder.layer.0.weight": rng.normal(
            size=(16, 32)
        ).astype(np.float32),
        "encoder.layer.1.weight": rng.normal(
            size=(8, 16)
        ).astype(np.float32),
        "encoder.layer.0.bias": np.ones(
            16, dtype=np.float32
        ),
    }

    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    return path


def test_inspect(checkpoint):
    result = inspect_model(checkpoint)

    assert result["total_tensors"] == 3
    assert result["eligible_matrices"] == 2
    assert result["layers"][0]["shape"] == [16, 32]


def test_analyze(checkpoint):
    result = analyze_model(
        checkpoint,
        max_layers=2,
        batch_size=4,
        group_size=16,
    )

    assert result["analyzed_matrices"] == 2

    for layer in result["layers"]:
        assert "FP32" in layer["methods"]
        assert "INT8" in layer["methods"]
        assert "INT4" in layer["methods"]
        assert "INT4_GROUP_16" in layer["methods"]


def test_reproducible(checkpoint):
    first = analyze_model(checkpoint, max_layers=1)
    second = analyze_model(checkpoint, max_layers=1)

    assert first["layers"] == second["layers"]


@pytest.mark.parametrize("group", [0, 1, 3])
def test_invalid_group(checkpoint, group):
    with pytest.raises(ValueError):
        analyze_model(checkpoint, group_size=group)


def test_error_metric():
    a = np.array([[1, 2]], dtype=np.float32)

    assert relative_error_pct(a, a) == 0.0


def test_invalid_extension(tmp_path):
    path = tmp_path / "model.npy"
    path.write_bytes(b"invalid")

    with pytest.raises(ValueError):
        inspect_model(path)
