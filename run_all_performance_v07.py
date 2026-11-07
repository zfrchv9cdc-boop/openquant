
"""OpenQuant: adaptive pytest and all-category benchmark runner."""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
PYTHON = sys.executable

CPU_MODULES = {
    "test_checkpoint.py",
    "test_gpu_decode.py",
    "test_groupwise.py",
    "test_model_eval.py",
    "test_quality_metrics_v07.py",
    "test_quantization.py",
    "test_report.py",
    "test_tuned_runtime.py",
}

STRATEGY_FILE = REPORTS / "test_strategy_v07.json"
runs = []


def run(label, arguments, env=None):
    command = [PYTHON, *arguments]
    logfile = REPORTS / f"test_run_{label}.log"

    print("\n" + "=" * 72)
    print("TASK:", label)
    print("COMMAND:", subprocess.list2cmdline(command))
    print("=" * 72, flush=True)

    start = time.perf_counter()
    output = []

    with logfile.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            output.append(line)

        code = process.wait()

    elapsed = time.perf_counter() - start
    transcript = "".join(output)

    passed = sum(
        int(number)
        for number in re.findall(r"(\d+) passed", transcript)
    )
    skipped = sum(
        int(number)
        for number in re.findall(r"(\d+) skipped", transcript)
    )

    result = {
        "label": label,
        "elapsed_seconds": round(elapsed, 3),
        "exit_code": code,
        "passed": passed,
        "skipped": skipped,
        "log": str(logfile),
    }
    runs.append(result)

    print(
        f"\nFINISHED {label}: {elapsed:.3f}s "
        f"(exit={code}, passed={passed}, skipped={skipped})"
    )

    if code != 0:
        save_report()
        raise SystemExit(
            f"{label} failed. Full output: {logfile}"
        )

    return result


def save_report():
    output = REPORTS / "all_performance_v07.json"
    output.write_text(
        json.dumps(
            {"runs": runs},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return output


def discover_tests():
    all_files = sorted(
        (ROOT / "tests").glob("test_*.py")
    )

    if not all_files:
        raise RuntimeError("No pytest files discovered")

    cpu = [
        str(path.relative_to(ROOT))
        for path in all_files
        if path.name in CPU_MODULES
    ]

    # Unknown/new modules default to serial GPU lane.
    gpu = [
        str(path.relative_to(ROOT))
        for path in all_files
        if path.name not in CPU_MODULES
    ]

    print(
        f"Discovered {len(all_files)} modules: "
        f"{len(cpu)} CPU, {len(gpu)} GPU/mixed."
    )

    return cpu, gpu


def pytest_run(label, files, workers=0):
    (REPORTS / "pytest_temp").mkdir(
        parents=True, exist_ok=True
    )

    args = [
        "-m", "pytest",
        "-q", "-x",
        "--durations=15",
        "--basetemp",
        str(REPORTS / "pytest_temp" / label),
    ]

    environment = os.environ.copy()

    if workers:
        args.extend([
            "-n", str(workers),
            "--dist", "loadfile",
        ])

        # Prevent excessive CPU threads per worker.
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            environment[key] = "1"

    args.extend(files)

    return run(label, args, environment)


def baseline():
    return pytest_run("full_serial", ["tests"])


def calibrate():
    if importlib.util.find_spec("xdist") is None:
        raise SystemExit(
            "Install pytest-xdist first."
        )

    cpu, gpu = discover_tests()

    options = []

    for workers in (0, 2, 4):
        label = f"cpu_workers_{workers}"
        result = pytest_run(label, cpu, workers)
        options.append((workers, result))

    best_workers, best_cpu = min(
        options,
        key=lambda item: item[1]["elapsed_seconds"],
    )

    gpu_result = pytest_run("gpu_serial", gpu)
    full_result = baseline()

    split_time = (
        best_cpu["elapsed_seconds"]
        + gpu_result["elapsed_seconds"]
    )

    split_passed = (
        best_cpu["passed"] + gpu_result["passed"]
    )
    split_skipped = (
        best_cpu["skipped"] + gpu_result["skipped"]
    )

    if (
        split_passed != full_result["passed"]
        or split_skipped != full_result["skipped"]
    ):
        raise RuntimeError(
            "Split and full test counts differ. "
            "Keep the original full test runner."
        )

    choice = (
        "split"
        if split_time < full_result["elapsed_seconds"]
        else "full"
    )

    strategy = {
        "choice": choice,
        "cpu_workers": best_workers,
        "full_seconds": full_result["elapsed_seconds"],
        "split_seconds": round(split_time, 3),
        "passed": full_result["passed"],
        "skipped": full_result["skipped"],
        "cpu_trials": [
            {
                "workers": workers,
                "seconds": result["elapsed_seconds"],
            }
            for workers, result in options
        ],
    }

    STRATEGY_FILE.write_text(
        json.dumps(strategy, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nCALIBRATION RESULT")
    print(json.dumps(strategy, indent=2))
    print("Saved:", STRATEGY_FILE)

    return strategy


def fast():
    if not STRATEGY_FILE.exists():
        print("No calibration: using complete serial suite.")
        return baseline()

    strategy = json.loads(
        STRATEGY_FILE.read_text(encoding="utf-8")
    )

    if strategy["choice"] == "full":
        print("Full serial suite measured faster.")
        return baseline()

    cpu, gpu = discover_tests()
    workers = strategy["cpu_workers"]

    cpu_result = pytest_run(
        "fast_cpu",
        cpu,
        workers,
    )
    gpu_result = pytest_run(
        "fast_gpu",
        gpu,
        0,
    )

    if (
        cpu_result["passed"] + gpu_result["passed"]
        != strategy["passed"]
    ):
        raise RuntimeError(
            "Test counts changed: rerun calibration."
        )

    print("\nAdaptive test execution: PASS")


def benchmarks():
    # CPU precision, packing and reconstruction.
    run(
        "cpu_quantization",
        ["-m", "openquant.benchmark"],
    )

    # Shared CUDA process for all twelve format/batch cases.
    run(
        "gpu_formats_matrix",
        [
            "benchmark_gpu_formats_matrix_v07.py",
            "--repeats", "20",
        ],
    )

    # Benchmark the installed production INT8 kernel.
    run(
        "fused_int8_production",
        ["benchmark_fused_int8_v07.py"],
    )

    # Existing fused INT4 kernel benchmark.
    run(
        "triton_kernel_speed",
        ["-m", "openquant.kernel_speed_benchmark"],
    )

    # Wider configuration matrix, including batched GEMV.
    run(
        "kernel_sweep_full",
        ["tune_all_shapes_v07.py", "--full", "--rep", "10"],
    )

    # End-to-end sentence encoder latency.
    run(
        "encoder_api",
        ["benchmark_api_v07.py"],
    )

    # Mixed-shape graph admission costs.
    run(
        "graph_admission",
        ["benchmark_admission_balanced_v07.py"],
    )

    # Cache limits and eager fallbacks.
    for budget in (16, 80, 256):
        run(
            f"graph_budget_{budget}",
            [
                "benchmark_graph_budget_v07.py",
                "--budget-mb", str(budget),
            ],
        )

    # Full STS-B: not a reduced sample.
    run(
        "semantic_quality_full",
        [
            "benchmark_stsb_fast_v07.py",
            "--limit", "0",
            "--batch", "32",
        ],
    )

    quality_path = REPORTS / "stsb_fast_v07.json"

    if quality_path.exists():
        quality = json.loads(
            quality_path.read_text(encoding="utf-8")
        )
        metrics = quality["metrics"]

        print("\nFULL QUALITY RESULTS")
        print("Pairs:", quality["pairs"])
        print("INT4 Spearman:", metrics["int4_spearman"])
        print("INT4 Pearson:", metrics["int4_pearson"])
        print(
            "Spearman drop (points):",
            metrics["spearman_drop_points"],
        )

    # Build and validate the Python distribution.
    run(
        "package_build",
        ["-m", "build", "--wheel", "--no-isolation"],
    )

    wheels = sorted(
        (ROOT / "dist").glob("*.whl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not wheels:
        raise RuntimeError("Build produced no wheel")

    run(
        "wheel_validation",
        [
            "-m", "twine", "check",
            str(wheels[0]),
        ],
    )

    run("dependency_check", ["-m", "pip", "check"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "calibrate",
            "fast",
            "benchmarks",
            "all",
        ),
        default="fast",
    )

    args = parser.parse_args()

    try:
        if args.mode == "baseline":
            baseline()
        elif args.mode == "calibrate":
            calibrate()
        elif args.mode == "fast":
            fast()
        elif args.mode == "benchmarks":
            benchmarks()
        else:
            calibrate()
            benchmarks()
    finally:
        output = save_report()
        print("\nReport:", output.resolve())


if __name__ == "__main__":
    main()
