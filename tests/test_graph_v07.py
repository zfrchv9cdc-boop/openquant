
"""Regression tests for OpenQuant CUDA Graph inference."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from benchmark_graph_v07 import GraphedSentenceEncoder
from openquant.inference_v07 import encode_sentences


class TinyEncoder(nn.Module):
    """Small deterministic encoder for graph correctness tests."""

    def __init__(self):
        super().__init__()

        self.embeddings = nn.Embedding(64, 16)
        self.projection = nn.Linear(16, 16)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        x = self.embeddings(input_ids)
        x = self.projection(x)

        return SimpleNamespace(
            last_hidden_state=x
        )


def create_example(batch, sequence):
    ids = torch.randint(
        low=1,
        high=63,
        size=(batch, sequence),
        device="cuda",
    )

    mask = torch.ones_like(ids)

    return {
        "input_ids": ids,
        "attention_mask": mask,
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required",
)
@pytest.mark.parametrize(
    "batch,sequence",
    [(1, 8), (3, 12)],
)
def test_graph_matches_eager(
    batch, sequence
):
    torch.manual_seed(42)

    model = TinyEncoder().cuda().eval()
    tokens = create_example(batch, sequence)

    graph = GraphedSentenceEncoder(
        model, tokens
    )

    with torch.inference_mode():
        reference = encode_sentences(
            model, tokens
        ).clone()

        actual = graph(tokens).clone()

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        reference,
        rtol=0.001,
        atol=0.001,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required",
)
def test_graph_changed_input():
    model = TinyEncoder().cuda().eval()

    tokens = create_example(2, 10)

    graph = GraphedSentenceEncoder(
        model, tokens
    )

    changed = {
        key: value.clone()
        for key, value in tokens.items()
    }

    changed["input_ids"][0, 2] = 63

    with torch.inference_mode():
        expected = encode_sentences(
            model, changed
        ).clone()

        actual = graph(changed).clone()

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.001,
        atol=0.001,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required",
)
def test_graph_rejects_shape_change():
    model = TinyEncoder().cuda().eval()

    tokens = create_example(2, 8)

    graph = GraphedSentenceEncoder(
        model, tokens
    )

    wrong_shape = create_example(3, 8)

    with pytest.raises(ValueError, match="shape"):
        graph(wrong_shape)
