from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"


def flatten_metrics(run_name: str, split_name: str, metrics: Dict[str, float]) -> Dict[str, float]:
    row: Dict[str, float] = {"run": run_name, "split": split_name}
    row.update(metrics)
    return row


def main() -> None:
    rows: List[Dict[str, float]] = []

    for run_dir in sorted(RESULTS_ROOT.glob("*final")) + sorted(RESULTS_ROOT.glob("ablation_*")):
        run_name = run_dir.name
        summary_path = run_dir / "training_summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        dataset = summary.get("dataset", "unknown")
        val_metrics = summary.get("validation_metrics", {})
        test_metrics = summary.get("test_metrics", {})

        if val_metrics:
            row = flatten_metrics(run_name, "validation", val_metrics)
            row["dataset"] = dataset
            rows.append(row)
        if test_metrics:
            row = flatten_metrics(run_name, "test", test_metrics)
            row["dataset"] = dataset
            rows.append(row)

    if not rows:
        raise SystemExit("No completed final runs found.")

    output_dir = PROJECT_ROOT / "paper_assets" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_runs_metrics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved comparison CSV to {output_path}")


if __name__ == "__main__":
    main()
