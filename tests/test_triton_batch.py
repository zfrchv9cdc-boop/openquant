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
    "m,n,k,group,block_n",
    [
        (1, 17, 65, 16, 4),
        (2, 33, 79, 16, 8),
        (3, 129, 255, 32, 8),
        (4, 1536, 384, 32, 4),
        (4, 1536, 384, 32, 8),
        (4, 1536, 384, 32, 16),
    ],
)
def test_batched_gemv(m, n, k, group, block_n):
    rng = np.random.default_rng(123)

    w = rng.normal(size=(n, k)).astype(np.float32)
    x = rng.normal(size=(m, k)).astype(np.float32)

    q = quantize_int4_groupwise(w, group)

    x_gpu = torch.from_numpy(x).cuda().half()
    data = torch.from_numpy(q.data).cuda()
    scales = torch.from_numpy(q.scales).cuda()

    output = int4_linear(
        x_gpu, data, scales, n, group,
        algorithm="batched",
        block_n=block_n,
    )

    reconstructed = torch.from_numpy(
        dequantize_int4_groupwise(q)
    ).cuda().half()

    reference = (x_gpu @ reconstructed.T).float()

    assert output.shape == reference.shape

    torch.testing.assert_close(
        output, reference, rtol=0.02, atol=0.02
    )
