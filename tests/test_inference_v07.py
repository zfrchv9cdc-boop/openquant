
"""OpenQuant v0.7 packed inference correctness tests."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("triton")

from torch import nn

from openquant.groupwise import (
    quantize_int4_groupwise,
    dequantize_int4_groupwise,
)

from openquant.inference_v07 import (
    PackedInt4Embedding,
    PackedInt4Linear,
)


def make_embedding():
    rng = np.random.default_rng(2027)
    weights = rng.normal(size=(11, 65)).astype(np.float32)

    q = quantize_int4_groupwise(weights, 16)

    original = nn.Embedding(11, 65)

    module = PackedInt4Embedding(
        original, q.data, q.scales, q.shape, 16
    )

    return module, q


def test_embedding_decodes_selected_rows():
    module, q = make_embedding()

    ids = torch.tensor([[0, 3, 10], [5, 5, 2]])
    actual = module(ids)

    restored = torch.from_numpy(
        dequantize_int4_groupwise(q)
    )

    expected = restored[ids]

    torch.testing.assert_close(
        actual, expected, rtol=1e-5, atol=1e-5
    )


def test_embedding_output_shape():
    module, _ = make_embedding()

    ids = torch.tensor([[[1, 2], [3, 4]]])

    assert module(ids).shape == (1, 2, 2, 65)


def test_embedding_rejects_bad_packed_size():
    _, q = make_embedding()

    original = nn.Embedding(11, 65)

    with pytest.raises(ValueError, match="packed"):
        PackedInt4Embedding(
            original,
            q.data[:-1],
            q.scales,
            q.shape,
            16,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required",
)
def test_packed_linear_gpu_matches_reference():
    rng = np.random.default_rng(42)

    weights = rng.normal(
        size=(17, 65)
    ).astype(np.float32)

    q = quantize_int4_groupwise(weights, 16)

    original = nn.Linear(65, 17, bias=True)

    with torch.no_grad():
        original.weight.copy_(
            torch.from_numpy(weights)
        )
        original.bias.zero_()

    packed_layer = PackedInt4Linear(
        original, q.data, q.scales, q.shape, 16
    ).cuda()

    x = torch.randn(
        (2, 65),
        device="cuda",
        dtype=torch.float32,
    )

    actual = packed_layer(x)

    restored = torch.from_numpy(
        dequantize_int4_groupwise(q)
    ).cuda().half()

    expected = (
        x.half() @ restored.T
    ).float()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.02,
        atol=0.02,
    )
