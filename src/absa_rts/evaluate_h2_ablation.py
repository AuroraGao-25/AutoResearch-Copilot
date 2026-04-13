from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from absa_rts.evaluate_baselines import evaluate_model, load_eval_rows, load_pairs, load_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate H2 ablation predictions (A1/C1/C2/C4) on ABSA-RTS."
    )
    parser.add_argument("--eval-csv", required=True, help="ABSA-RTS flattened evaluation CSV")
    parser.add_argument("--pairs-csv", required=True, help="ABSA-RTS pairs CSV")
    parser.add_argument("--a1-predictions", required=True, help="A1 prediction file (CSV or JSONL)")
    parser.add_argument("--c1-predictions", required=True, help="C1 prediction file (CSV or JSONL)")
    parser.add_argument("--c2-predictions", required=True, help="C2 prediction file (CSV or JSONL)")
    parser.add_argument("--c4-predictions", required=True, help="C4 prediction file (CSV or JSONL)")
    parser.add_argument("--output-dir", required=True, help="Output directory for report files")
    return parser.parse_args()


def _delta(a: float, b: float) -> float:
    return b - a


def build_summary_rows(metrics_by_model: dict[str, dict[str, object]]) -> list[dict[str, str]]:
    base = metrics_by_model["A1"]
    rows: list[dict[str, str]] = []
    for model_name in ["A1", "C1", "C2", "C4"]:
        m = metrics_by_model[model_name]
        row = {
            "model": model_name,
            "accuracy": f"{m['accuracy']:.4f}",
            "macro_f1": f"{m['macro_f1']:.4f}",
            "metamorphic_pass_rate": f"{m['metamorphic_pass_rate']:.4f}",
            "delta_macro_f1_vs_a1": f"{_delta(base['macro_f1'], m['macro_f1']):+.4f}",
            "delta_pass_vs_a1": f"{_delta(base['metamorphic_pass_rate'], m['metamorphic_pass_rate']):+.4f}",
        }
        rows.append(row)
    return rows


def build_category_rows(metrics_by_model: dict[str, dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for category in ["negation", "contrast"]:
        a1_cat = metrics_by_model["A1"]["category_metrics"].get(category, {})
        for model_name in ["A1", "C1", "C2", "C4"]:
            m_cat = metrics_by_model[model_name]["category_metrics"].get(category, {})
            row = {
                "category": category,
                "model": model_name,
                "accuracy": f"{m_cat.get('accuracy', 0.0):.4f}",
                "macro_f1": f"{m_cat.get('macro_f1', 0.0):.4f}",
                "metamorphic_pass_rate": f"{m_cat.get('metamorphic_pass_rate', 0.0):.4f}",
                "delta_pass_vs_a1": f"{_delta(a1_cat.get('metamorphic_pass_rate', 0.0), m_cat.get('metamorphic_pass_rate', 0.0)):+.4f}",
            }
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    eval_rows = load_eval_rows(Path(args.eval_csv))
    pairs = load_pairs(Path(args.pairs_csv))

    prediction_files = {
        "A1": args.a1_predictions,
        "C1": args.c1_predictions,
        "C2": args.c2_predictions,
        "C4": args.c4_predictions,
    }
    metrics_by_model: dict[str, dict[str, object]] = {}
    for model_name, path in prediction_files.items():
        preds = load_predictions(Path(path))
        metrics_by_model[model_name] = evaluate_model(
            model_name=model_name,
            eval_rows=eval_rows,
            pairs=pairs,
            predictions=preds,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "h2_metrics.json").write_text(
        json.dumps({"models": [metrics_by_model[k] for k in ["A1", "C1", "C2", "C4"]]}, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "h2_summary.csv", build_summary_rows(metrics_by_model))
    write_csv(out_dir / "h2_category_summary.csv", build_category_rows(metrics_by_model))
    print(f"Wrote H2 ablation reports to {out_dir}")


if __name__ == "__main__":
    main()

