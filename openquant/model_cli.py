import json
from pathlib import Path

from openquant.model_eval import inspect_model, analyze_model


def configure_commands(commands):
    inspect = commands.add_parser(
        "inspect-model",
        help="Inspect pretrained Safetensors weights",
    )
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--limit", type=int, default=12)
    inspect.add_argument("--contains", default="")

    analyze = commands.add_parser(
        "analyze-model",
        help="Compare INT8 and INT4 on pretrained weights",
    )
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--max-layers", type=int, default=6)
    analyze.add_argument("--batch-size", type=int, default=8)
    analyze.add_argument("--group-size", type=int, default=32)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--contains", default="")
    analyze.add_argument(
        "--output",
        default="reports/real_model_v06.json",
    )


def run_inspect(args):
    info = inspect_model(
        args.input,
        limit=args.limit,
        contains=args.contains,
    )

    print("Checkpoint tensors:", info["total_tensors"])
    print("Eligible matrices:", info["eligible_matrices"])

    for layer in info["layers"]:
        print(
            layer["name"],
            layer["shape"],
            layer["dtype"],
        )


def run_analyze(args):
    data = analyze_model(
        args.input,
        max_layers=args.max_layers,
        batch_size=args.batch_size,
        group_size=args.group_size,
        seed=args.seed,
        contains=args.contains,
    )

    for layer in data["layers"]:
        print("\nLAYER:", layer["name"])
        print("SHAPE:", layer["shape"])

        for method, metrics in layer["methods"].items():
            print(
                f"{method:<16}"
                f"saved={metrics['memory_reduction_pct']:>7.2f}% "
                f"weight_error={metrics['weight_error_pct']:>8.3f}% "
                f"output_error={metrics['output_error_pct']:>8.3f}%"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nReport saved:", output.resolve())
