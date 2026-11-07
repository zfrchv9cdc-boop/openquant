
"""OpenQuant portable INT8 checkpoint demonstration."""

import json
from pathlib import Path

import torch
from torch import nn

from openquant.torch_int8 import convert_linear_layers_int8
from openquant.int8_checkpoint_v07 import (
    save_int8_checkpoint,
    load_int8_checkpoint,
)


def architecture():
    return nn.Sequential(
        nn.Linear(384, 768),
        nn.GELU(),
        nn.Linear(768, 384),
    ).eval()


def main():
    torch.manual_seed(42)

    output = Path(
        "reports/demo_portable_int8.safetensors"
    )

    # Preserve old runs rather than overwriting them.
    if output.exists() or output.with_suffix(".json").exists():
        raise SystemExit(
            "Demo checkpoint exists. Use a new filename "
            "or rename the previous demo files."
        )

    original = architecture()

    packed, report = convert_linear_layers_int8(
        original, min_parameters=1024
    )

    x = torch.randn(8, 384)

    with torch.inference_mode():
        before = packed(x)

    manifest = save_int8_checkpoint(
        packed, output
    )

    # A newly constructed architecture has unrelated
    # random initialized weights. The checkpoint restores
    # the actual compressed model state.
    fresh_skeleton = architecture()

    restored, _ = load_int8_checkpoint(
        fresh_skeleton,
        output,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    with torch.inference_mode():
        after = restored(x.to(device)).cpu()

    torch.testing.assert_close(
        after, before,
        rtol=0.02,
        atol=0.05,
    )

    print("\nPORTABLE INT8 CHECKPOINT")
    print("-" * 55)
    print("Device:", device)
    print("Converted layers:", report["converted_layers"])
    print(
        "Weight saving:",
        round(
            report["weight_storage_saved_percent"],
            2,
        ),
        "%",
    )
    print("Checkpoint size:", output.stat().st_size, "bytes")
    print("Format:", manifest["format"])
    print("Output shape:", tuple(after.shape))
    print("Reload correctness: PASS")

    summary = {
        "checkpoint": str(output),
        "converted_layers": report["converted_layers"],
        "compressed_bytes": output.stat().st_size,
        "output_shape": list(after.shape),
        "correctness": "PASS",
    }

    report_path = Path(
        "reports/portable_int8_demo.json"
    )

    report_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Report:", report_path.resolve())


if __name__ == "__main__":
    main()
