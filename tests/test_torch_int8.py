
"""Tests for model-independent INT8 linear conversion."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from openquant.torch_int8 import (
    PackedInt8Linear,
    convert_linear_layers_int8,
)


def make_model():
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    ).eval()


def relative_error(actual, reference):
    return (
        torch.linalg.vector_norm(actual.float() - reference.float())
        / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    ).item()


@pytest.mark.parametrize(
    "shape",
    [
        (4, 32),
        (2, 3, 32),
        (1, 1, 32),
    ],
)
def test_nested_model_preserves_output_shape(shape):
    model = make_model()
    x = torch.randn(shape)

    expected = model(x)
    compressed, report = convert_linear_layers_int8(
        model, min_parameters=0,
    )
    actual = compressed(x)

    assert actual.shape == expected.shape
    assert report["converted_layers"] == 2
    assert relative_error(actual, expected) < 0.05


def test_original_model_is_unchanged():
    model = make_model()

    converted, report = convert_linear_layers_int8(
        model, min_parameters=0,
    )

    assert converted is not model
    assert type(model[0]) is nn.Linear
    assert isinstance(converted[0], PackedInt8Linear)
    assert report["converted_layers"] == 2


def test_weight_memory_is_reduced():
    model = make_model()

    _, report = convert_linear_layers_int8(
        model, min_parameters=0,
    )

    assert report["int8_weight_bytes"] < report["original_weight_bytes"]
    assert report["weight_storage_saved_percent"] > 60


def test_inplace_conversion():
    model = make_model()

    converted, report = convert_linear_layers_int8(
        model, min_parameters=0, inplace=True,
    )

    assert converted is model
    assert isinstance(model[0], PackedInt8Linear)
    assert report["converted_layers"] == 2


def test_exclude_named_layer():
    model = make_model()

    converted, report = convert_linear_layers_int8(
        model,
        min_parameters=0,
        exclude=("0",),
    )

    assert type(converted[0]) is nn.Linear
    assert isinstance(converted[2], PackedInt8Linear)
    assert report["converted_layers"] == 1


def test_small_layers_are_skipped():
    model = nn.Sequential(
        nn.Linear(2, 2),
    ).eval()

    converted, report = convert_linear_layers_int8(
        model,
        min_parameters=1024,
    )

    assert type(converted[0]) is nn.Linear
    assert report["converted_layers"] == 0


def test_bias_preserved():
    layer = nn.Linear(32, 16).eval()
    x = torch.zeros(4, 32)

    reference = layer(x)
    compressed = PackedInt8Linear(layer)
    actual = compressed(x)

    torch.testing.assert_close(actual, reference)


def test_without_bias():
    layer = nn.Linear(32, 16, bias=False).eval()

    x = torch.randn(4, 32)
    actual = PackedInt8Linear(layer)(x)
    expected = layer(x)

    assert relative_error(actual, expected) < 0.05


def test_noncontiguous_input():
    layer = nn.Linear(32, 16).eval()

    x = torch.randn(3, 5, 32).transpose(0, 1)
    assert not x.is_contiguous()

    output = PackedInt8Linear(layer)(x)
    assert output.shape == (5, 3, 16)


def test_zero_length_batch():
    layer = nn.Linear(32, 16).eval()

    x = torch.empty(0, 32)
    output = PackedInt8Linear(layer)(x)

    assert output.shape == (0, 16)


def test_training_rejected():
    layer = PackedInt8Linear(
        nn.Linear(32, 16).eval()
    )
    layer.train()

    with pytest.raises(RuntimeError, match="inference-only"):
        layer(torch.randn(4, 32))


def test_invalid_input_dimension():
    layer = PackedInt8Linear(
        nn.Linear(32, 16).eval()
    )

    with pytest.raises(ValueError):
        layer(torch.randn(4, 31))


def test_model_must_be_in_eval_mode():
    model = make_model()
    model.train()

    with pytest.raises(ValueError):
        convert_linear_layers_int8(
            model, min_parameters=0,
        )


def test_state_dict_roundtrip():
    model = make_model()

    first, _ = convert_linear_layers_int8(
        model, min_parameters=0,
    )
    second, _ = convert_linear_layers_int8(
        model, min_parameters=0,
    )

    second.load_state_dict(first.state_dict())

    x = torch.randn(4, 32)

    torch.testing.assert_close(
        first(x),
        second(x),
    )


def test_shared_module_alias():
    class SharedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(32, 16)
            self.second = self.first

        def forward(self, x):
            return self.first(x) + self.second(x)

    model = SharedModel().eval()
    x = torch.randn(3, 32)
    reference = model(x)

    converted, report = convert_linear_layers_int8(
        model, min_parameters=0,
    )

    assert converted.first is converted.second
    assert report["converted_layers"] == 1
    assert relative_error(converted(x), reference) < 0.05


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_cuda_fused_execution():
    model = make_model().cuda()
    x = torch.randn(4, 32, device="cuda")

    with torch.inference_mode():
        reference = model(x)

        compressed, report = convert_linear_layers_int8(
            model, min_parameters=0,
        )

        actual = compressed(x)

    assert report["converted_layers"] == 2
    assert actual.is_cuda
    assert actual.shape == reference.shape
    assert relative_error(actual, reference) < 0.05
