
"""Run real sentence encoding using OpenQuant."""

import json
import torch

from openquant.encoder_v07 import OpenQuantEncoder


def main():
    encoder = OpenQuantEncoder(
        batch_size=4,
        max_length=128,
        max_graphs=2,
    )

    sentences = [
        "Artificial intelligence is changing computing.",
        "Machine learning allows computers to learn patterns.",
        "The weather is sunny today.",
        "The sky is clear and the weather is pleasant.",
    ]

    first = encoder.encode(sentences)
    second = encoder.encode(sentences)

    print("Embedding shape:", tuple(first.shape))
    print("Embedding dtype:", first.dtype)

    print("\nCosine similarity matrix:")
    print(
        torch.mm(first.float(), first.float().T).round(
            decimals=4
        )
    )

    torch.testing.assert_close(
        first, second, rtol=0.001, atol=0.001
    )

    print("\nGraph replay correctness: PASS")

    print("\nGraph cache:")
    print(json.dumps(
        encoder.cache_info(),
        indent=2,
    ))


if __name__ == "__main__":
    main()
