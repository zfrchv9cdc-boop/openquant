
"""Portable INT8 checkpoint tests."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from torch import nn
from safetensors.torch import load_file

from openquant.torch_int8 import (
    PackedInt8Linear,
    convert_linear_layers_int8,
)
from openquant.int8_checkpoint_v07 import (
    load_int8_checkpoint,
    save_int8_checkpoint,
)


def model():
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    ).eval()


def compressed():
    return convert_linear_layers_int8(
        model(), min_parameters=0
    )[0]


def test_cpu_roundtrip(tmp_path):
    torch.manual_seed(42)
    x = torch.randn(4, 32)

    original = compressed()
    reference = original(x)

    file = tmp_path / "network.safetensors"
    save_int8_checkpoint(original, file)

    restored, info = load_int8_checkpoint(
        model(), file
    )

    torch.testing.assert_close(
        restored(x),
        reference,
        rtol=0,
        atol=0,
    )
    assert info["format"] == "openquant-torch-int8-v1"
    assert info["compressed_paths"] == 2


def test_original_skeleton_preserved(tmp_path):
    file = tmp_path / "network.safetensors"
    save_int8_checkpoint(compressed(), file)

    skeleton = model()
    restored, _ = load_int8_checkpoint(
        skeleton, file
    )

    assert type(skeleton[0]) is nn.Linear
    assert isinstance(restored[0], PackedInt8Linear)
    assert restored is not skeleton


def test_inplace_restore(tmp_path):
    file = tmp_path / "network.safetensors"
    save_int8_checkpoint(compressed(), file)

    skeleton = model()
    restored, _ = load_int8_checkpoint(
        skeleton, file, inplace=True
    )

    assert restored is skeleton
    assert isinstance(skeleton[0], PackedInt8Linear)


@pytest.mark.parametrize(
    "bad",
    [
        nn.Sequential(
            nn.Linear(31, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        ).eval(),
        nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 16),
        ).eval(),
    ],
)
def test_incompatible_architecture_rejected(tmp_path, bad):
    file = tmp_path / "network.safetensors"
    save_int8_checkpoint(compressed(), file)

    with pytest.raises(ValueError):
        load_int8_checkpoint(bad, file)


def test_corrupted_payload_rejected(tmp_path):
    file = tmp_path / "network.safetensors"
    save_int8_checkpoint(compressed(), file)

    with file.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        load_int8_checkpoint(model(), file)


def test_existing_checkpoint_protected(tmp_path):
    file = tmp_path / "network.safetensors"
    source = compressed()

    save_int8_checkpoint(source, file)

    with pytest.raises(FileExistsError):
        save_int8_checkpoint(source, file)


def test_wrong_format_rejected(tmp_path):
    file = tmp_path / "network.safetensors"
    save_int8_checkpoint(compressed(), file)

    manifest_file = file.with_suffix(".json")
    data = json.loads(manifest_file.read_text())
    data["format"] = "invalid"
    manifest_file.write_text(json.dumps(data))

    with pytest.raises(
        ValueError,
        match="Unsupported checkpoint format",
    ):
        load_int8_checkpoint(model(), file)


def test_quantized_layers_have_no_fp32_weights(tmp_path):
    file = tmp_path / "network.safetensors"
    save_int8_checkpoint(compressed(), file)

    tensors = load_file(str(file))

    assert "0.packed" in tensors
    assert "0.scales" in tensors
    assert "2.packed" in tensors
    assert "2.scales" in tensors

    assert "0.weight" not in tensors
    assert "2.weight" not in tensors
    assert tensors["0.packed"].dtype == torch.int8
    assert tensors["2.packed"].dtype == torch.int8


def test_partial_conversion_preserves_raw_layers(tmp_path):
    original = model()

    converted, _ = convert_linear_layers_int8(
        original,
        min_parameters=1500,
    )

    assert isinstance(converted[0], PackedInt8Linear)
    assert type(converted[2]) is nn.Linear

    x = torch.randn(4, 32)
    reference = converted(x)

    file = tmp_path / "mixed.safetensors"
    save_int8_checkpoint(converted, file)

    tensors = load_file(str(file))

    assert "0.packed" in tensors
    assert "2.weight" in tensors

    restored, _ = load_int8_checkpoint(
        model(), file
    )

    torch.testing.assert_close(
        restored(x),
        reference,
        rtol=0,
        atol=0,
    )


def test_root_linear_roundtrip(tmp_path):
    original = nn.Linear(32, 16).eval()

    compressed_root, _ = convert_linear_layers_int8(
        original, min_parameters=0
    )

    file = tmp_path / "root.safetensors"
    save_int8_checkpoint(compressed_root, file)

    restored, _ = load_int8_checkpoint(
        nn.Linear(32, 16).eval(), file
    )

    x = torch.randn(2, 32)

    assert isinstance(restored, PackedInt8Linear)

    torch.testing.assert_close(
        restored(x),
        compressed_root(x),
        rtol=0,
        atol=0,
    )


def test_shared_alias_restored(tmp_path):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(32, 16)
            self.second = self.first

        def forward(self, x):
            return self.first(x) + self.second(x)

    source, _ = convert_linear_layers_int8(
        Shared().eval(), min_parameters=0
    )

    file = tmp_path / "shared.safetensors"
    save_int8_checkpoint(source, file)

    restored, _ = load_int8_checkpoint(
        Shared().eval(), file
    )

    assert restored.first is restored.second

    x = torch.randn(2, 32)

    torch.testing.assert_close(
        restored(x),
        source(x),
        rtol=0,
        atol=0,
    )


def test_uncompressed_model_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="No PackedInt8Linear",
    ):
        save_int8_checkpoint(
            model(),
            tmp_path / "empty.safetensors",
        )


def test_invalid_extension_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_int8_checkpoint(
            compressed(),
            tmp_path / "model.pt",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_cuda_restore(tmp_path):
    source = compressed()

    file = tmp_path / "gpu.safetensors"
    save_int8_checkpoint(source, file)

    restored, _ = load_int8_checkpoint(
        model(),
        file,
        device="cuda",
    )

    assert restored[0].packed.is_cuda

    x_cpu = torch.randn(4, 32)
    x_gpu = x_cpu.cuda()

    with torch.inference_mode():
        expected = source(x_cpu)
        actual = restored(x_gpu).cpu()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.02,
        atol=0.05,
    )
