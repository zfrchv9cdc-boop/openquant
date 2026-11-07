"""CPU-only tests for the sentence encoding CLI."""

import argparse
import json

import pytest

from openquant import cli
from openquant.encode_cli import run_encode


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


class FakeEncoder:
    received_options = None
    received_encode = None

    def __init__(self, **options):
        type(self).received_options = options

    def encode(self, texts, **options):
        type(self).received_encode = options
        return FakeTensor([
            [float(index), 0.5, 1.0]
            for index in range(len(texts))
        ])


def options(tmp_path, **overrides):
    values = {
        "text": ["hello"],
        "input": None,
        "output": tmp_path / "embeddings.json",
        "archive": tmp_path / "model.safetensors",
        "batch_size": 12,
        "max_length": 128,
        "graphs": False,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_single_sentence(tmp_path):
    args = options(tmp_path)
    assert run_encode(
        args, encoder_factory=FakeEncoder
    ) == 0

    data = json.loads(
        args.output.read_text(encoding="utf-8")
    )
    assert data["count"] == 1
    assert data["dimensions"] == 3
    assert data["sentences"] == ["hello"]
    assert len(data["embeddings"]) == 1


def test_multiple_sentences(tmp_path):
    args = options(
        tmp_path,
        text=["hello", "world", "again"],
    )
    run_encode(args, encoder_factory=FakeEncoder)
    data = json.loads(args.output.read_text())
    assert data["count"] == 3
    assert len(data["embeddings"]) == 3


def test_input_file(tmp_path):
    path = tmp_path / "sentences.txt"
    path.write_text(
        "First sentence\n\nSecond sentence\n",
        encoding="utf-8",
    )
    args = options(
        tmp_path,
        text=None,
        input=path,
    )
    run_encode(args, encoder_factory=FakeEncoder)
    data = json.loads(args.output.read_text())
    assert data["sentences"] == [
        "First sentence",
        "Second sentence",
    ]


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n  \n", encoding="utf-8")
    args = options(tmp_path, text=None, input=path)
    with pytest.raises(ValueError):
        run_encode(args, encoder_factory=FakeEncoder)


@pytest.mark.parametrize(
    "setting,value",
    [
        ("batch_size", 0),
        ("max_length", 3),
        ("max_length", 513),
    ],
)
def test_invalid_configurations(tmp_path, setting, value):
    args = options(tmp_path, **{setting: value})
    with pytest.raises(ValueError):
        run_encode(args, encoder_factory=FakeEncoder)


def test_existing_output_protected(tmp_path):
    args = options(tmp_path)
    args.output.write_text("preserve")
    with pytest.raises(FileExistsError):
        run_encode(args, encoder_factory=FakeEncoder)
    assert args.output.read_text() == "preserve"


def test_overwrite_is_explicit(tmp_path):
    args = options(tmp_path, overwrite=True)
    args.output.write_text("old")
    run_encode(args, encoder_factory=FakeEncoder)
    assert json.loads(
        args.output.read_text()
    )["count"] == 1


def test_archive_protected(tmp_path):
    args = options(tmp_path)
    args.output = args.archive
    with pytest.raises(ValueError):
        run_encode(args, encoder_factory=FakeEncoder)


def test_graph_option_is_forwarded(tmp_path):
    args = options(tmp_path, graphs=True)
    run_encode(args, encoder_factory=FakeEncoder)
    assert FakeEncoder.received_options[
        "enable_graphs"
    ] is True
    assert FakeEncoder.received_encode[
        "use_graph"
    ] is True


def test_cli_dispatches_without_cuda(tmp_path, monkeypatch):
    import openquant.encode_cli as command

    received = []
    monkeypatch.setattr(
        command,
        "run_encode",
        lambda args: received.append(args.text) or 0,
    )

    result = cli.main([
        "encode",
        "--text", "hello",
        "--output", str(tmp_path / "out.json"),
    ])

    assert result == 0
    assert received == [["hello"]]


def test_cli_requires_source(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "encode",
            "--output", str(tmp_path / "out.json"),
        ])
    assert exc.value.code == 2
