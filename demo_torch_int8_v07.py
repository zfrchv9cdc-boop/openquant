
"""Model-independent INT8 inference demonstration."""

import torch
from torch import nn

from openquant.torch_int8 import (
    convert_linear_layers_int8,
)


def main():
    torch.manual_seed(42)

    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = nn.Sequential(
        nn.Linear(384, 768),
        nn.GELU(),
        nn.Linear(768, 384),
    ).to(device).eval()

    x = torch.randn(8, 384, device=device)

    with torch.inference_mode():
        reference = model(x)

        compressed, report = convert_linear_layers_int8(
            model,
            min_parameters=1024,
        )

        result = compressed(x)

    error = (
        torch.linalg.vector_norm(
            result.float() - reference.float()
        )
        / torch.linalg.vector_norm(
            reference.float()
        ).clamp_min(1e-12)
        * 100
    ).item()

    print("Device:", device)
    print("Original model preserved:", isinstance(model[0], nn.Linear))
    print("Converted layers:", report["converted_layers"])
    print(
        "Original converted weights:",
        report["original_weight_bytes"],
        "bytes",
    )
    print(
        "Compressed weights:",
        report["int8_weight_bytes"],
        "bytes",
    )
    print(
        "Weight memory saved:",
        round(report["weight_storage_saved_percent"], 2),
        "%",
    )
    print("Output shape:", tuple(result.shape))
    print("Relative output error:", round(error, 4), "%")

    assert result.shape == reference.shape
    assert report["converted_layers"] == 2
    assert error < 5.0

    print("MODEL INTEROPERABILITY: PASS")


if __name__ == "__main__":
    main()
