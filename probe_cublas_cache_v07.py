
"""Test whether cuBLAS workspaces explain warmup memory growth."""

import json
import torch
from pathlib import Path

from probe_graph_lifecycle_v07 import TinyModel
from probe_warmup_stream_v07 import (
    capture_once,
    memory,
    tokens_for,
)
from openquant.inference_v07 import encode_sentences


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    clear_cublas = getattr(
        torch._C,
        "_cuda_clearCublasWorkspaces",
        None,
    )

    if not callable(clear_cublas):
        raise RuntimeError(
            "cuBLAS workspace cleanup unavailable "
            "in this PyTorch installation"
        )

    torch.manual_seed(42)
    model = TinyModel().cuda().eval()

    with torch.inference_mode():
        example = tokens_for(16)
        output = encode_sentences(model, example)
        del output, example

    baseline = memory()
    rows = []

    print("GPU:", torch.cuda.get_device_name(0))
    print("Baseline:", baseline, "MiB")

    for cycle in range(1, 9):
        for length in (16, 32, 64, 16):
            capture_once(model, length, "side")

        before = memory()

        # All graphs created by capture_once have been
        # reset and released before this diagnostic call.
        torch.cuda.synchronize()
        clear_cublas()
        after = memory()

        row = {
            "cycle": cycle,
            "before_mib": before,
            "after_mib": after,
            "reclaimed_mib": round(
                before - after, 3
            ),
        }
        rows.append(row)

        print(
            f"Cycle {cycle}: "
            f"before={before:.3f}, "
            f"after={after:.3f}, "
            f"reclaimed={row['reclaimed_mib']:.3f} MiB"
        )

    drift = round(
        rows[-1]["after_mib"] - rows[0]["after_mib"],
        3,
    )

    report = {
        "baseline_mib": baseline,
        "after_cleanup_drift_mib": drift,
        "rows": rows,
    }

    output = Path("reports/cublas_cache_probe_v07.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nAfter-cleanup drift:", drift, "MiB")
    print("Report:", output.resolve())


if __name__ == "__main__":
    main()
