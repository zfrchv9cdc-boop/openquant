
"""OpenQuant v0.7 reusable encoder regression tests."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from openquant.encoder_v07 import OpenQuantEncoder


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
        sequences = []

        for text in texts:
            tokens = (
                [1]
                + [4 + ord(c) % 70 for c in text]
                + [2]
            )

            tokens = tokens[:max_length]
            sequences.append(tokens)

        length = max(map(len, sequences))

        ids = torch.zeros(
            (len(texts), length),
            dtype=torch.long,
        )

        mask = torch.zeros_like(ids)

        for i, tokens in enumerate(sequences):
            ids[i, :len(tokens)] = torch.tensor(tokens)
            mask[i, :len(tokens)] = 1

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


@pytest.fixture
def encoder():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")

    torch.manual_seed(42)

    return OpenQuantEncoder(
        model=TinyModel(),
        tokenizer=TinyTokenizer(),
        batch_size=3,
        max_length=64,
        max_graphs=2,
    )


def test_text_encoding_shape_and_norm(encoder):
    output = encoder.encode([
        "machine learning",
        "quantization",
    ])

    assert output.shape == (2, 16)

    norms = torch.linalg.vector_norm(output, dim=1)

    torch.testing.assert_close(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
        rtol=1e-5,
    )


def test_graph_matches_eager(encoder):
    texts = ["cat", "dog"]

    eager = encoder.encode(
        texts, use_graph=False
    )

    graphed = encoder.encode(
        texts, use_graph=True
    )

    torch.testing.assert_close(
        eager, graphed, rtol=0.001, atol=0.001
    )

    assert encoder.cache_info()["cached_graphs"] == 1


def test_graph_replay_changes_with_input(encoder):
    first = encoder.encode(["cat", "dog"])

    # The shape is unchanged, so this should reuse
    # the same captured graph with different token IDs.
    second = encoder.encode(["zzz", "dog"])

    assert not torch.allclose(first[0], second[0])

    info = encoder.cache_info()

    assert info["stats"].get("graph_hits", 0) >= 1


def test_cache_limit_and_batching():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")

    encoder = OpenQuantEncoder(
        model=TinyModel(),
        tokenizer=TinyTokenizer(),
        batch_size=3,
        max_length=64,
        max_graphs=1,
    )

    short = encoder.encode(["cat"])
    long = encoder.encode([
        "This is a considerably longer sentence."
    ])

    assert short.shape == (1, 16)
    assert long.shape == (1, 16)

    # A second shape cannot create an unlimited graph.
    assert encoder.cache_info()["cached_graphs"] == 1

    # Seven items must be processed in three chunks.
    output = encoder.encode(
        ["cat"] * 7,
        use_graph=False,
    )

    assert output.shape == (7, 16)

    encoder.clear_graph_cache()

    assert encoder.cache_info()["cached_graphs"] == 0
