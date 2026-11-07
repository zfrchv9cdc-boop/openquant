
"""Verify shared OpenQuant encoder correctness under concurrency."""

from concurrent.futures import ThreadPoolExecutor

import torch

from openquant.encoder_v07 import OpenQuantEncoder


REQUESTS = [
    ["A small neural network.", "A tiny neural model."],
    ["The weather is sunny.", "It is raining outside."],
    ["Quantization reduces memory.", "FP32 uses more bytes."],
    ["The spacecraft reached orbit.", "A rocket entered space."],
] * 4


def main():
    encoder = OpenQuantEncoder(
        batch_size=4,
        max_graphs=2,
        max_length=128,
    )

    # Establish independent expected outputs.
    expected = [
        encoder.encode(request, use_graph=False)
        for request in REQUESTS
    ]

    # Force creation of the first graph before concurrency.
    encoder.encode(REQUESTS[0], use_graph=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(
            pool.map(
                lambda request: encoder.encode(
                    request,
                    use_graph=True,
                ),
                REQUESTS,
            )
        )

    for index, (reference, result) in enumerate(
        zip(expected, actual)
    ):
        torch.testing.assert_close(
            result,
            reference,
            rtol=0.001,
            atol=0.001,
        )

        assert torch.isfinite(result).all(), index

    info = encoder.cache_info()

    assert info["cached_graphs"] <= 2

    print("Concurrent requests:", len(REQUESTS))
    print("Correctness: PASS")
    print("Cached graphs:", info["cached_graphs"])
    print("Statistics:", info["stats"])


if __name__ == "__main__":
    main()
