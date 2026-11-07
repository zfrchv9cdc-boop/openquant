"""Production fused INT8 correctness and validation tests."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from openquant.quantization import quantize, dequantize
from openquant.triton_int8 import int8_linear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required",
)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 17, 65),
        (4, 33, 79),
        (16, 384, 384),
        (128, 1536, 384),
        (1, 1536, 384),
        (4, 1536, 384),
        (8, 32, 64),
        (1, 1, 1),
    ],
)
def test_int8_linear_matches_reference(m, n, k):
    rng = np.random.default_rng(42)

    weights = rng.normal(
        size=(n, k)
    ).astype(np.float32)

    q = quantize(weights, bits=8)

    packed = torch.from_numpy(q.data).cuda()
    scales = torch.from_numpy(q.scales).cuda()

    x = torch.from_numpy(
        rng.normal(size=(m, k)).astype(np.float32)
    ).cuda().half().contiguous()

    reference_weights = torch.from_numpy(
        dequantize(q)
    ).cuda()

    reference = x.float() @ reference_weights.T

    actual = int8_linear(x, packed, scales)

    torch.cuda.synchronize()

    assert actual.shape == (m, n)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()

    torch.testing.assert_close(
        actual,
        reference,
        rtol=0.02,
        atol=0.05,
    )


def test_int8_linear_zero_weights():
    weights = np.zeros((33, 79), dtype=np.float32)
    q = quantize(weights, bits=8)

    packed = torch.from_numpy(q.data).cuda()
    scales = torch.from_numpy(q.scales).cuda()

    x = torch.randn(
        (4, 79),
        device="cuda",
        dtype=torch.float16,
    )

    actual = int8_linear(x, packed, scales)

    torch.testing.assert_close(
        actual,
        torch.zeros_like(actual),
    )


@pytest.mark.parametrize(
    "failure",
    [
        "input_dtype",
        "weight_dtype",
        "scale_dtype",
        "dimension",
        "scale_count",
        "noncontiguous",
        "empty_batch",
    ],
)
def test_int8_linear_rejects_bad_inputs(failure):
    x = torch.randn(
        (4, 64),
        device="cuda",
        dtype=torch.float16,
    )

    packed = torch.ones(
        (32, 64),
        device="cuda",
        dtype=torch.int8,
    )

    scales = torch.ones(
        (32, 1),
        device="cuda",
        dtype=torch.float32,
    )

    if failure == "input_dtype":
        x = x.float()

    elif failure == "weight_dtype":
        packed = packed.to(torch.int16)

    elif failure == "scale_dtype":
        scales = scales.half()

    elif failure == "dimension":
        packed = packed[:, :-1].contiguous()

    elif failure == "scale_count":
        scales = scales[:-1].contiguous()

    elif failure == "noncontiguous":
        x = torch.randn(
            (64, 4),
            device="cuda",
            dtype=torch.float16,
        ).T

    elif failure == "empty_batch":
        x = x[:0]

    with pytest.raises(ValueError):
        int8_linear(x, packed, scales)