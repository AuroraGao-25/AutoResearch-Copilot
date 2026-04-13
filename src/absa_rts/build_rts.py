from __future__ import annotations

import argparse
import csv
from pathlib import Path

from absa_rts.transforms import make_contrast_pair, make_negation_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ABSA-RTS metamorphic pairs from seed aspect/sentiment data."
    )
    parser.add_argument("--input", required=True, help="Seed CSV path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--domain",
        default="restaurants",
        help="Domain label written into output files.",
    )
    return parser.parse_args()


def read_seed_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "aspect", "sentiment"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return [row for row in reader]


def build_pairs(rows: list[dict[str, str]], domain: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    pairs: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []

    pair_index = 0
    for row in rows:
        base_id = row["id"].strip()
        aspect = row["aspect"]
        sentiment = row["sentiment"]

        pair_candidates = []
        neg_pair = make_negation_pair(aspect=aspect, sentiment=sentiment)
        if neg_pair is not None:
            pair_candidates.append(neg_pair)
        pair_candidates.append(make_contrast_pair(aspect=aspect, sentiment=sentiment))

        for pair in pair_candidates:
            pair_index += 1
            pair_id = f"{base_id}_{pair.category}_{pair_index:04d}"
            src_case_id = f"{pair_id}_src"
            fup_case_id = f"{pair_id}_fup"

            pairs.append(
                {
                    "pair_id": pair_id,
                    "source_id": base_id,
                    "domain": domain,
                    "category": pair.category,
                    "relation_type": pair.relation_type,
                    "aspect": pair.aspect,
                    "source_case_id": src_case_id,
                    "followup_case_id": fup_case_id,
                    "source_text": pair.source_text,
                    "followup_text": pair.followup_text,
                    "source_sentiment": pair.source_sentiment,
                    "expected_followup_sentiment": pair.followup_sentiment,
                }
            )
            eval_rows.extend(
                [
                    {
                        "case_id": src_case_id,
                        "pair_id": pair_id,
                        "domain": domain,
                        "category": pair.category,
                        "relation_type": pair.relation_type,
                        "is_followup": "0",
                        "aspect": pair.aspect,
                        "text": pair.source_text,
                        "gold_sentiment": pair.source_sentiment,
                    },
                    {
                        "case_id": fup_case_id,
                        "pair_id": pair_id,
                        "domain": domain,
                        "category": pair.category,
                        "relation_type": pair.relation_type,
                        "is_followup": "1",
                        "aspect": pair.aspect,
                        "text": pair.followup_text,
                        "gold_sentiment": pair.followup_sentiment,
                    },
                ]
            )
    return pairs, eval_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_path = Path(args.input)
    output_dir = Path(args.output_dir)

    rows = read_seed_rows(seed_path)
    pairs, eval_rows = build_pairs(rows, args.domain.strip().lower())

    write_csv(output_dir / "absa_rts_pairs.csv", pairs)
    write_csv(output_dir / "absa_rts_eval.csv", eval_rows)

    print(f"Generated {len(pairs)} pairs and {len(eval_rows)} evaluation rows in {output_dir}")


if __name__ == "__main__":
    main()

