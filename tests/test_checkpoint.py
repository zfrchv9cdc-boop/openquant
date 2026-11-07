import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from openquant.checkpoint import pack_checkpoint, unpack_checkpoint


@pytest.fixture
def sample(tmp_path):
    rng = np.random.default_rng(17)
    original = {
        "layer.weight": rng.normal(size=(16, 64)).astype(np.float32),
        "other.weight": rng.normal(size=(16, 64)).astype(np.float16),
        "layer.bias": np.ones(16, dtype=np.float32),
    }
    path = tmp_path / "source.safetensors"
    save_file(original, str(path))
    return path, original


def test_round_trip(sample, tmp_path):
    source, original = sample
    packed = tmp_path / "packed.safetensors"
    restored = tmp_path / "restored.safetensors"

    stats = pack_checkpoint(
        source, packed, group_size=16, min_elements=1
    )
    assert stats["quantized_tensors"] == 2
    assert packed.with_suffix(".json").exists()

    result = unpack_checkpoint(packed, restored)
    assert result["restored_tensors"] == 3

    actual = load_file(str(restored))
    assert set(actual) == set(original)

    for name, value in original.items():
        assert actual[name].shape == value.shape
        assert actual[name].dtype == value.dtype

        if name.endswith("bias"):
            np.testing.assert_array_equal(actual[name], value)
        else:
            a = actual[name].astype(np.float32)
            b = value.astype(np.float32)
            relative_error = np.linalg.norm(a - b) / np.linalg.norm(b)
            assert relative_error < 0.2


def test_pack_cannot_overwrite(sample, tmp_path):
    source, _ = sample
    target = tmp_path / "packed.safetensors"
    pack_checkpoint(source, target)

    with pytest.raises(FileExistsError):
        pack_checkpoint(source, target)


def test_invalid_group_size(sample, tmp_path):
    source, _ = sample

    with pytest.raises(ValueError):
        pack_checkpoint(
            source,
            tmp_path / "output.safetensors",
            group_size=3,
        )


def test_small_weights_remain_raw(sample, tmp_path):
    source, _ = sample
    packed = tmp_path / "packed.safetensors"
    restored = tmp_path / "restored.safetensors"

    result = pack_checkpoint(
        source, packed, min_elements=100000
    )
    assert result["quantized_tensors"] == 0

    unpack_checkpoint(packed, restored)

    original = load_file(str(source))
    actual = load_file(str(restored))

    for name, array in original.items():
        np.testing.assert_array_equal(actual[name], array)
