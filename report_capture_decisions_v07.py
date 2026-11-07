"""Summarize OpenQuant capture and memory reports."""

import json
from pathlib import Path

ROOT = Path("reports")


def read(path):
    if not path.is_file():
        print("MISSING:", path)
        return None

    return json.loads(
        path.read_text(encoding="utf-8")
    )


print("\nCAPTURE POLICY COMPARISON")

for budget, floor in [
    (16, 0),
    (16, 32),
    (80, 32),
    (128, 32),
]:
    path = ROOT / (
        f"capture_floor_budget{budget}"
        f"_floor{floor}.json"
    )

    record = read(path)

    if record is None:
        continue

    summary = record["summary"]
    events = summary["event_totals"]

    print(
        f"budget={budget:3} MiB "
        f"floor={floor:2} MiB | "
        f"median={summary['median_total_ms']:9.3f} ms | "
        f"captures={events['graph_captures']:2} | "
        f"rejected={events['budget_rejections']:2} | "
        f"preflight={events['preflight_budget_fallbacks']:2}"
    )


print("\nMEMORY AFTER GRAPH CACHE CLEAR")

for mode in ("eager", "immediate", "two-hit"):
    path = ROOT / (
        f"memory_stability_v07_{mode}.json"
    )

    record = read(path)

    if record is None:
        continue

    summary = record["summary"]

    print(
        f"{mode:10} | "
        f"median-after-clear="
        f"{summary['median_after_clear_allocated_mib']:9.3f} MiB | "
        f"first-to-last="
        f"{summary['after_clear_first_to_last_delta_mib']:+9.3f} MiB | "
        f"highest-peak="
        f"{summary['max_peak_allocated_mib']:9.3f} MiB"
    )


print("\nSummary complete.")