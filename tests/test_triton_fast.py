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
@pytest.mark.parametrize(
    "algorithm,m,n,k,group",
    [
        ("gemv", 1, 17, 65, 16),
        ("gemv", 4, 129, 255, 32),
        ("packed", 16, 33, 79, 16),
        ("packed", 128, 128, 256, 32),
    ],
)
def test_accelerated_int4(algorithm, m, n, k, group):
    rng = np.random.default_rng(42)

    weights = rng.normal(size=(n, k)).astype(np.float32)
    inputs = rng.normal(size=(m, k)).astype(np.float32)

    q = quantize_int4_groupwise(weights, group)

    x = torch.from_numpy(inputs).cuda().half()
    data = torch.from_numpy(q.data).cuda()
    scales = torch.from_numpy(q.scales).cuda()

    result = int4_linear(
        x, data, scales, n, group,
        algorithm=algorithm,
    )

    restored = torch.from_numpy(
        dequantize_int4_groupwise(q)
    ).cuda().half()

    reference = (x @ restored.T).float()

    assert result.shape == (m, n)
    assert torch.isfinite(result).all()

    torch.testing.assert_close(
        result,
        reference,
        rtol=0.02,
        atol=0.02,
    )
