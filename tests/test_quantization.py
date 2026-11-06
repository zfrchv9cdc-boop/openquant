import numpy as np
import pytest

from openquant.quantization import quantize, dequantize


@pytest.mark.parametrize("bits,limit", [
    (8, 0.02),
    (4, 0.20),
])
def test_reconstruction(bits, limit):
    rng = np.random.default_rng(42)
    weights = rng.normal(size=(128, 256)).astype(np.float32)

    result = quantize(weights, bits)
    restored = dequantize(result)

    assert restored.shape == weights.shape

    error = np.linalg.norm(weights - restored) / np.linalg.norm(weights)
    assert error < limit


@pytest.mark.parametrize("bits", [4, 8])
def test_zeros_and_odd_shapes(bits):
    weights = np.array([
        [0, 0, 0],
        [1, -1, 0.5],
        [0, -0.5, 1],
    ], dtype=np.float32)

    result = quantize(weights, bits)
    restored = dequantize(result)

    assert restored.shape == (3, 3)
    assert np.all(np.isfinite(restored))
    assert np.all(restored[0] == 0)


@pytest.mark.parametrize("bits", [4, 8])
def test_storage(bits):
    weights = np.ones((17, 19), dtype=np.float32)
    result = quantize(weights, bits)

    count = weights.size
    expected = count if bits == 8 else (count + 1) // 2

    assert result.data.nbytes == expected
    assert result.data.nbytes + result.scales.nbytes < weights.nbytes


def test_invalid_precision():
    with pytest.raises(ValueError):
        quantize(np.ones((2, 2)), bits=3)


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        quantize(np.ones(10), bits=8)
