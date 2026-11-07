import json
from pathlib import Path

ROOT = Path("reports")

modes = (
    "direct",
    "direct-early",
    "fixed",
    "fixed-early",
    "encoder",
)

print("\nOPENQUANT GRAPH LIFECYCLE COMPARISON")
print("-" * 76)

for mode in modes:
    path = ROOT / f"graph_layers_{mode}.json"

    if not path.exists():
        print(f"{mode:15} MISSING")
        continue

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    cycles = data["cycles"]
    baseline = data["baseline"]["allocated_mib"]
    first = cycles[0]["allocated_mib"]
    last = cycles[-1]["allocated_mib"]

    print(
        f"{mode:15} "
        f"baseline={baseline:9.3f} MiB | "
        f"first={first:9.3f} MiB | "
        f"last={last:9.3f} MiB | "
        f"drift={data['drift_mib']:+9.3f} MiB"
    )

print("-" * 76)