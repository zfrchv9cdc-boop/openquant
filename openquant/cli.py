from openquant.model_cli import configure_commands, run_inspect, run_analyze
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="openquant",
        description="OpenQuant AI optimization toolkit",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="OpenQuant 0.6.0.dev0",
    )

    commands = parser.add_subparsers(dest="command")

    cpu = commands.add_parser("cpu")
    cpu.add_argument("--rows", type=int, default=1024)
    cpu.add_argument("--cols", type=int, default=1024)

    commands.add_parser("gpu")
    commands.add_parser("tune")

    report = commands.add_parser(
        "report",
        help="Save a reproducible CPU quantization report",
    )

    report.add_argument("--rows", type=int, default=1024)
    report.add_argument("--cols", type=int, default=1024)
    report.add_argument("--seed", type=int, default=42)
    report.add_argument("--group-size", type=int, default=32)
    report.add_argument(
        "--output",
        default="reports/openquant_cpu.json",
    )

    configure_commands(commands)

    args = parser.parse_args(argv)

    if args.command == "cpu":
        if args.rows <= 0 or args.cols <= 0:
            parser.error("Dimensions must be positive")

        from openquant.benchmark import benchmark
        benchmark(args.rows, args.cols)

    elif args.command == "gpu":
        from openquant.fused_benchmark import main as run
        run()

    elif args.command == "tune":
        from openquant.autotune import main as run
        run()

    elif args.command == "inspect-model":
        run_inspect(args)

    elif args.command == "analyze-model":
        run_analyze(args)

    elif args.command == "report":
        from openquant.report import make_cpu_report

        try:
            data = make_cpu_report(
                rows=args.rows,
                cols=args.cols,
                seed=args.seed,
                group_size=args.group_size,
            )
        except ValueError as exc:
            parser.error(str(exc))

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        output.write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        print("Report saved:", output.resolve())

        for name, metrics in data["formats"].items():
            print(
                f"{name:<16}"
                f"{metrics['memory_reduction_pct']:>8.2f}% saved"
                f"  error: "
                f"{metrics['relative_weight_error_pct']:.3f}%"
            )

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
