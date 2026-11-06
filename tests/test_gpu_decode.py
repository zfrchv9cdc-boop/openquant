import numpy as np
import pytest
import torch

from openquant.quantization import quantize, dequantize
from openquant.gpu_benchmark import decode_gpu


@pytest.mark.parametrize("bits,shape", [
    (4, (5, 7)),
    (8, (5, 7)),
    (4, (1, 1)),
    (8, (1, 1)),
    (4, (3, 1)),
])
def test_decode_matches_numpy(bits, shape):
    rng = np.random.default_rng(100)

    weights = rng.standard_normal(
        shape
    ).astype(np.float32)

    q = quantize(weights, bits)

    result = decode_gpu(
        torch.from_numpy(q.data),
        torch.from_numpy(q.scales),
        bits,
        shape
    )

    np.testing.assert_allclose(
        result.numpy(),
        dequantize(q),
        rtol=0,
        atol=1e-6
    )
