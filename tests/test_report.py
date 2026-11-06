import json

import pytest

from openquant.report import make_cpu_report
from openquant.cli import main


@pytest.mark.parametrize("rows,cols,group", [
    (3, 5, 4),
    (4, 7, 16),
])
def test_report_shapes(rows, cols, group):
    result = make_cpu_report(
        rows, cols, seed=7, group_size=group
    )

    assert result["weights"]["shape"] == [rows, cols]

    for name in (
        "FP32", "INT8", "INT4", f"INT4_GROUP_{group}"
    ):
        assert name in result["formats"]
        assert result["formats"][name]["storage_bytes"] > 0


def test_report_reproducibility():
    first = make_cpu_report(16, 32, seed=42)
    second = make_cpu_report(16, 32, seed=42)

    assert first["formats"] == second["formats"]


@pytest.mark.parametrize("rows,cols,group", [
    (0, 8, 4),
    (3, 0, 4),
    (2, 3, 3),
])
def test_invalid_parameters(rows, cols, group):
    with pytest.raises(ValueError):
        make_cpu_report(rows, cols, group_size=group)


def test_cli_json_output(tmp_path):
    output = tmp_path / "report.json"

    status = main([
        "report",
        "--rows", "8",
        "--cols", "16",
        "--group-size", "8",
        "--output", str(output),
    ])

    assert status == 0
    assert output.exists()

    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert "INT4_GROUP_8" in data["formats"]
