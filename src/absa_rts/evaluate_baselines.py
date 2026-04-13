from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from absa_rts.simple_model import predict_sentiment


LABELS = ("negative", "neutral", "positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ABSA baselines on ABSA-RTS and compute metamorphic pass rate."
    )
    parser.add_argument("--eval-csv", required=True, help="ABSA-RTS flattened evaluation CSV.")
    parser.add_argument("--pairs-csv", required=True, help="ABSA-RTS pair metadata CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for metric outputs.")
    parser.add_argument(
        "--a0-predictions",
        help="Optional CSV/JSONL with case_id,pred_sentiment. If omitted, a simple rule-based predictor is used.",
    )
    parser.add_argument(
        "--a1-predictions",
        required=True,
        help="CSV/JSONL with case_id,pred_sentiment for A1.",
    )
    return parser.parse_args()


def load_eval_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"case_id", "pair_id", "category", "relation_type", "text", "gold_sentiment", "is_followup"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Missing columns in eval CSV: {sorted(missing)}")
    return rows


def load_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"pair_id", "category", "relation_type", "source_case_id", "followup_case_id", "expected_followup_sentiment"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Missing columns in pairs CSV: {sorted(missing)}")
    return rows


def _normalize_sentiment(value: str) -> str:
    sentiment = value.strip().lower()
    if sentiment not in LABELS:
        raise ValueError(f"Unsupported sentiment label: '{value}'")
    return sentiment


def load_predictions(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".jsonl":
        predictions: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                case_id = obj["case_id"]
                pred = obj.get("pred_sentiment")
                if pred is None and isinstance(obj.get("prediction"), dict):
                    pred = obj["prediction"].get("sentiment")
                if pred is None:
                    raise ValueError("JSONL prediction row missing sentiment")
                predictions[case_id] = _normalize_sentiment(pred)
        return predictions

    predictions = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            predictions[row["case_id"]] = _normalize_sentiment(row["pred_sentiment"])
    return predictions


def build_a0_predictions(eval_rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["case_id"]: predict_sentiment(row["text"]) for row in eval_rows}


def accuracy(golds: list[str], preds: list[str]) -> float:
    if not golds:
        return 0.0
    return sum(g == p for g, p in zip(golds, preds)) / len(golds)


def macro_f1(golds: list[str], preds: list[str]) -> float:
    if not golds:
        return 0.0
    per_label_f1 = []
    for label in LABELS:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        per_label_f1.append(f1)
    return sum(per_label_f1) / len(per_label_f1)


def metamorphic_pass_rate(pairs: list[dict[str, str]], preds: dict[str, str]) -> tuple[float, dict[str, float]]:
    total = 0
    passed = 0
    category_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for pair in pairs:
        src = preds.get(pair["source_case_id"])
        fup = preds.get(pair["followup_case_id"])
        if src is None or fup is None:
            continue
        relation = pair["relation_type"].strip().lower()
        expected_fup = _normalize_sentiment(pair["expected_followup_sentiment"])

        if relation == "invariant":
            ok = src == fup
        elif relation == "flip":
            ok = fup == expected_fup
        else:
            raise ValueError(f"Unsupported relation_type '{relation}'")

        total += 1
        category = pair["category"]
        category_counts[category][0] += 1
        if ok:
            passed += 1
            category_counts[category][1] += 1

    overall = (passed / total) if total else 0.0
    by_category = {
        category: (counts[1] / counts[0] if counts[0] else 0.0)
        for category, counts in category_counts.items()
    }
    return overall, by_category


def evaluate_model(
    model_name: str,
    eval_rows: list[dict[str, str]],
    pairs: list[dict[str, str]],
    predictions: dict[str, str],
) -> dict[str, object]:
    missing = [row["case_id"] for row in eval_rows if row["case_id"] not in predictions]
    if missing:
        raise ValueError(f"{model_name} missing predictions for {len(missing)} cases")

    golds = [_normalize_sentiment(row["gold_sentiment"]) for row in eval_rows]
    preds = [predictions[row["case_id"]] for row in eval_rows]

    category_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in eval_rows:
        category_rows[row["category"]].append(
            (_normalize_sentiment(row["gold_sentiment"]), predictions[row["case_id"]])
        )

    pass_overall, pass_by_category = metamorphic_pass_rate(pairs, predictions)
    metrics = {
        "model": model_name,
        "accuracy": accuracy(golds, preds),
        "macro_f1": macro_f1(golds, preds),
        "metamorphic_pass_rate": pass_overall,
        "category_metrics": {},
    }
    for category, rows in category_rows.items():
        c_golds = [x[0] for x in rows]
        c_preds = [x[1] for x in rows]
        metrics["category_metrics"][category] = {
            "accuracy": accuracy(c_golds, c_preds),
            "macro_f1": macro_f1(c_golds, c_preds),
            "metamorphic_pass_rate": pass_by_category.get(category, 0.0),
        }
    return metrics


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_csv(path: Path, metrics: list[dict[str, object]]) -> None:
    rows = []
    for item in metrics:
        rows.append(
            {
                "model": item["model"],
                "accuracy": f"{item['accuracy']:.4f}",
                "macro_f1": f"{item['macro_f1']:.4f}",
                "metamorphic_pass_rate": f"{item['metamorphic_pass_rate']:.4f}",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "macro_f1", "metamorphic_pass_rate"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    eval_rows = load_eval_rows(Path(args.eval_csv))
    pairs = load_pairs(Path(args.pairs_csv))

    a0_predictions = (
        load_predictions(Path(args.a0_predictions))
        if args.a0_predictions
        else build_a0_predictions(eval_rows)
    )
    a1_predictions = load_predictions(Path(args.a1_predictions))

    all_metrics = [
        evaluate_model("A0", eval_rows, pairs, a0_predictions),
        evaluate_model("A1", eval_rows, pairs, a1_predictions),
    ]
    output_dir = Path(args.output_dir)
    write_json(output_dir / "baseline_metrics.json", {"models": all_metrics})
    write_summary_csv(output_dir / "baseline_summary.csv", all_metrics)
    print(f"Wrote metrics to {output_dir}")


if __name__ == "__main__":
    main()

