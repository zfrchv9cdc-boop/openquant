import numpy as np
import pytest
import torch

from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)
from openquant.triton_int4 import int4_linear


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required",
)
@pytest.mark.parametrize("shape", [
    (3, 7, 65, 16),
    (16, 32, 128, 32),
    (33, 55, 79, 16),
])
def test_fused_int4(shape):
    batch, rows, cols, group = shape

    rng = np.random.default_rng(42)

    w = rng.normal(
        size=(rows, cols)
    ).astype(np.float32)

    x = rng.normal(
        size=(batch, cols)
    ).astype(np.float32)

    q = quantize_int4_groupwise(w, group)

    x_gpu = torch.from_numpy(x).cuda().half()

    data = torch.from_numpy(q.data).cuda()
    scales = torch.from_numpy(q.scales).cuda()

    result = int4_linear(
        x_gpu, data, scales, rows, group
    )

    reconstructed = torch.from_numpy(
        dequantize_int4_groupwise(q)
    ).cuda().half()

    reference = (
        x_gpu @ reconstructed.T
    ).float()

    assert result.shape == reference.shape
    assert torch.isfinite(result).all()

    torch.testing.assert_close(
        result,
        reference,
        rtol=0.02,
        atol=0.02,
    )
