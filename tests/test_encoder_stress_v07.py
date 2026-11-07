"""OpenQuant v0.7 GPU memory and concurrency tests."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from openquant.encoder_v07 import OpenQuantEncoder


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ):
        sequences = [
            (
                [1]
                + [4 + ord(c) % 60 for c in text]
                + [2]
            )[:max_length]
            for text in texts
        ]

        length = max(map(len, sequences))

        ids = torch.zeros(
            (len(texts), length),
            dtype=torch.long,
        )

        mask = torch.zeros_like(ids)

        for i, sequence in enumerate(sequences):
            size = len(sequence)

            ids[i, :size] = torch.tensor(
                sequence,
                dtype=torch.long,
            )

            mask[i, :size] = 1

        return {
            "input_ids": ids,
            "attention_mask": mask,
        }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(80, 16)
        self.linear = nn.Linear(16, 16)

    def forward(self, input_ids, attention_mask):
        hidden = self.linear(
            self.embedding(input_ids)
        )

        return SimpleNamespace(
            last_hidden_state=hidden
        )


def make_encoder(**overrides):
    options = {
        "model": TinyModel(),
        "tokenizer": TinyTokenizer(),
        "batch_size": 4,
        "max_length": 64,
        "max_graphs": 2,
        "max_graph_cache_mb": 512,
        "min_capture_free_mb": 0,
        "graph_capture_min_calls": 1,
    }

    options.update(overrides)

    return OpenQuantEncoder(**options)


def test_peak_memory_during_graph_replacement():
    torch.manual_seed(42)

    encoder = make_encoder(max_graphs=1)

    encoder.encode(["cat"], use_graph=True)

    expected = encoder.encode(
        ["z" * 40],
        use_graph=False,
    )

    torch.cuda.synchronize()

    allocated_before = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()

    result = encoder.encode(
        ["z" * 40],
        use_graph=True,
    )

    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()

    torch.testing.assert_close(
        result,
        expected,
        rtol=0.001,
        atol=0.001,
    )

    info = encoder.cache_info()

    assert peak >= allocated_before
    assert info["cached_graphs"] == 1

    assert (
        info["graph_cache_bytes"]
        <= info["graph_budget_bytes"]
    )

    assert info["stats"]["graph_captures"] == 2
    assert info["stats"]["graph_evictions"] >= 1


def test_concurrent_encoder_requests_match_eager():
    torch.manual_seed(42)

    encoder = make_encoder(
        graph_capture_min_calls=2,
    )

    requests = [
        ["cat"],
        ["x" * 20],
        ["z" * 40],
        ["dog"],
    ] * 4

    references = [
        encoder.encode(
            request,
            use_graph=False,
        )
        for request in requests
    ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(
            pool.map(encoder.encode, requests)
        )

    assert len(outputs) == len(references)

    for actual, expected in zip(
        outputs,
        references,
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.001,
            atol=0.001,
        )

    info = encoder.cache_info()

    assert info["cached_graphs"] <= 2

    assert (
        info["graph_cache_bytes"]
        <= info["graph_budget_bytes"]
    )

    assert (
        info["stats"].get(
            "cold_shape_fallbacks", 0
        ) >= 1
    )
