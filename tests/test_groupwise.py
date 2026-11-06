import numpy as np
import pytest
import torch

from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)
from openquant.group_benchmark import restore_groupwise_torch


@pytest.mark.parametrize("shape", [(3, 67), (5, 32), (1, 1)])
@pytest.mark.parametrize("group_size", [16, 32])
def test_roundtrip(shape, group_size):
    rng = np.random.default_rng(7)
    weights = rng.normal(size=shape).astype(np.float32)

    q = quantize_int4_groupwise(weights, group_size)
    restored = dequantize_int4_groupwise(q)

    assert restored.shape == shape
    assert np.isfinite(restored).all()

    gpu_style = restore_groupwise_torch(
        torch.from_numpy(q.data),
        torch.from_numpy(q.scales),
        shape,
        group_size
    ).numpy()

    np.testing.assert_allclose(
        gpu_style, restored, rtol=0, atol=1e-6
    )


def test_zero_weights():
    w = np.zeros((2, 35), dtype=np.float32)
    q = quantize_int4_groupwise(w)
    assert np.all(dequantize_int4_groupwise(q) == 0)


@pytest.mark.parametrize("group_size", [0, 1, 3])
def test_invalid_group_size(group_size):
    with pytest.raises(ValueError):
        quantize_int4_groupwise(
            np.ones((2, 8)), group_size
        )


def test_invalid_shape():
    with pytest.raises(ValueError):
        quantize_int4_groupwise(np.ones(8))
