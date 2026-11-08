
"""Smoke-test a clean OpenQuant wheel outside the repository."""

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist" / "clean_v07"
CHECKPOINT = (
    ROOT / "reports" / "demo_portable_int8.safetensors"
)


def run(args, *, cwd, env=None):
    print("\nRUN:", subprocess.list2cmdline(args), flush=True)
    subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=True,
    )


def main():
    wheels = sorted(
        DIST.glob("openquant_lab-0.7.0.dev0-*.whl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not wheels:
        raise FileNotFoundError(
            "Build the v0.7 wheel before running this smoke test"
        )

    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

        required = (
            "openquant/torch_int8.py",
            "openquant/triton_int8.py",
            "openquant/int8_checkpoint_v07.py",
            "openquant/checkpoint_info_v07.py",
            "openquant/encode_cli.py",
        )

        for module in required:
            if module not in names:
                raise RuntimeError(
                    f"Missing packaged module: {module}"
                )

        if any(
            "experiment_30pct" in name
            for name in names
        ):
            raise RuntimeError(
                "Local experimental code entered the wheel"
            )

        metadata_name = next(
            name for name in names
            if name.endswith(".dist-info/METADATA")
        )

        metadata = archive.read(
            metadata_name
        ).decode("utf-8")

        if "Version: 0.7.0.dev0" not in metadata:
            raise RuntimeError(
                "Wheel metadata version is inconsistent"
            )

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            "The portable INT8 demo checkpoint is missing"
        )

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        target = temporary / "installed"

        run(
            [
                sys.executable, "-m", "pip",
                "install", "--no-deps",
                "--target", str(target),
                str(wheel),
            ],
            cwd=temporary,
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)

        # Running from the temporary directory prevents
        # Python from finding the local source checkout.
        commands = [
            ["--version"],
            ["cpu", "--rows", "32", "--cols", "32"],
            ["checkpoint-info", "--input", str(CHECKPOINT)],
        ]

        for arguments in commands:
            run(
                [
                    sys.executable,
                    "-m",
                    "openquant.cli",
                    *arguments,
                ],
                cwd=temporary,
                env=env,
            )

    print("\nCLEAN INSTALLED-WHEEL SMOKE TEST: PASS")
    print("Wheel:", wheel.resolve())
    print("Version:", "0.7.0.dev0")
    print("Repository source dependency: NONE")


if __name__ == "__main__":
    main()
