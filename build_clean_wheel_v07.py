
"""Build and inspect OpenQuant from the committed Git tree."""

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist" / "clean_v07"


def command(*args, cwd=None):
    print("\nRUN:", subprocess.list2cmdline(args), flush=True)
    subprocess.run(
        list(args),
        cwd=cwd or ROOT,
        check=True,
    )


def main():
    DIST.mkdir(parents=True, exist_ok=True)

    print("Building committed Git HEAD only")

    command(
        "git", "rev-parse", "--short", "HEAD"
    )

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        source = temporary / "source"
        source.mkdir()

        archive = temporary / "source.zip"

        command(
            "git", "archive",
            "--format=zip",
            "-o", str(archive),
            "HEAD",
        )

        with zipfile.ZipFile(archive) as file:
            file.extractall(source)

        command(
            sys.executable,
            "-m", "build",
            "--wheel",
            "--no-isolation",
            "--outdir", str(DIST),
            cwd=source,
        )

    wheels = sorted(
        DIST.glob("*.whl"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    if not wheels:
        raise RuntimeError(
            "No wheel was produced"
        )

    latest = wheels[0]

    command(
        sys.executable,
        "-m", "twine", "check",
        str(latest),
    )

    with zipfile.ZipFile(latest) as wheel:
        members = wheel.namelist()

        required = [
            "openquant/torch_int8.py",
            "openquant/triton_int8.py",
            "openquant/encode_cli.py",
            "openquant/int8_checkpoint_v07.py",
        ]

        missing = [
            name for name in required
            if name not in members
        ]

        if missing:
            raise RuntimeError(
                f"Missing files from wheel: {missing}"
            )

        experiments = [
            name for name in members
            if "experiment_30pct" in name
        ]

        if experiments:
            raise RuntimeError(
                f"Experimental files found: {experiments}"
            )

        print("\nCLEAN WHEEL VALIDATION")
        print("Wheel:", latest)
        print("Required modules:", len(required))
        print("Untracked experiment modules: 0")
        print("Result: PASS")


if __name__ == "__main__":
    main()
