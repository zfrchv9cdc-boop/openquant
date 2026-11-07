
"""Additional correctness coverage for experimental fused INT8."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from openquant.quantization import quantize, dequantize

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 17, 65),
        (4, 33, 79),
        (16, 384, 384),
        (128, 1536, 384),
    ],
)
def test_fused_int8_handles_varied_shapes(m, n, k):
    from openquant.triton_int8 import int8_linear as fused_int8

    rng = np.random.default_rng(42)

    weights = rng.normal(
        size=(n, k)
    ).astype(np.float32)

    q = quantize(weights, bits=8)

    packed = torch.from_numpy(q.data).cuda()
    scales = torch.from_numpy(
        q.scales
    ).cuda().reshape(-1)

    x = torch.from_numpy(
        rng.normal(size=(m, k)).astype(np.float32)
    ).cuda().half()

    reference_weights = torch.from_numpy(
        dequantize(q)
    ).cuda()

    reference = x.float() @ reference_weights.T

    result = fused_int8(
        x, packed, scales
    )

    torch.cuda.synchronize()

    assert result.shape == (m, n)
    assert torch.isfinite(result).all()

    torch.testing.assert_close(
        result,
        reference,
        rtol=0.02,
        atol=0.05,
    )
