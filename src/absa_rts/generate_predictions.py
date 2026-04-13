from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from absa_rts.simple_model import predict_sentiment


LABELS = ["negative", "neutral", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline prediction CSV from ABSA-RTS eval file.")
    parser.add_argument("--eval-csv", required=True, help="ABSA-RTS eval CSV path.")
    parser.add_argument("--output-csv", required=True, help="Output predictions CSV path.")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Probability [0,1] of replacing a prediction with a random other label.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible noise.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.noise <= 1.0:
        raise ValueError("--noise must be between 0 and 1")

    random.seed(args.seed)
    rows = []
    with Path(args.eval_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pred = predict_sentiment(row["text"])
            if args.noise > 0 and random.random() < args.noise:
                alternatives = [label for label in LABELS if label != pred]
                pred = random.choice(alternatives)
            rows.append({"case_id": row["case_id"], "pred_sentiment": pred})

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "pred_sentiment"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()

