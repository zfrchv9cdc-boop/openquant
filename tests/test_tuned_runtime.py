
"""Tests for OpenQuant tuned-runtime profile validation."""

import json

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

import openquant.tuned_runtime as runtime


@pytest.fixture
def setup_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda device: "Mock RTX",
    )

    report = {
        "gpu": "Mock RTX",
        "weight_shape": [1536, 384],
        "group_size": 32,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "results": [
            {
                "batch_size": 16,
                "best": {
                    "name": "packed_32_64_64_4",
                    "latency_ms": 0.00655,
                    "config": {
                        "algorithm": "packed",
                        "block_m": 32,
                        "block_n": 64,
                        "block_k": 64,
                        "num_warps": 4,
                    },
                },
            },
        ],
    }

    path = tmp_path / "profile.json"

    def build():
        path.write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return runtime.TunedInt4Linear(
            torch.zeros(1, dtype=torch.uint8),
            torch.zeros(1, dtype=torch.float32),
            1536,
            384,
            32,
            path,
        )

    return report, build


def test_profile_selects_packed(setup_runtime):
    _, build = setup_runtime
    tuned = build()

    config = tuned.selected_config(16)

    assert config["algorithm"] == "packed"
    assert config["block_m"] == 32
    assert config["block_n"] == 64


def test_unknown_batch_uses_fallback(setup_runtime):
    _, build = setup_runtime
    tuned = build()

    assert tuned.selected_config(8) == {
        "algorithm": "auto"
    }


def test_runtime_dispatch(setup_runtime, monkeypatch):
    _, build = setup_runtime
    tuned = build()

    calls = []

    def fake_linear(x, packed, scales, n, group, **kwargs):
        calls.append(kwargs)
        return x

    monkeypatch.setattr(
        runtime,
        "int4_linear",
        fake_linear,
    )

    x = torch.zeros((16, 384), dtype=torch.float16)
    tuned(x)

    assert calls[0]["algorithm"] == "packed"
    assert calls[0]["block_m"] == 32


def test_wrong_gpu_rejected(setup_runtime):
    report, build = setup_runtime
    report["gpu"] = "Another GPU"

    with pytest.raises(ValueError, match="GPU mismatch"):
        build()


def test_invalid_kernel_config_rejected(setup_runtime):
    report, build = setup_runtime
    report["results"][0]["best"]["config"]["unsafe"] = True

    with pytest.raises(ValueError, match="Unsupported"):
        build()


def test_stale_triton_version_rejected(setup_runtime):
    report, build = setup_runtime
    report["triton_version"] = "0.0.0"

    with pytest.raises(ValueError, match="Triton version"):
        build()


def test_duplicate_batch_rejected(setup_runtime):
    report, build = setup_runtime
    report["results"].append(
        dict(report["results"][0])
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build()


def test_input_dimension_rejected(setup_runtime):
    _, build = setup_runtime
    tuned = build()

    bad_input = torch.zeros((16, 385))

    with pytest.raises(ValueError, match="dimension"):
        tuned(bad_input)
