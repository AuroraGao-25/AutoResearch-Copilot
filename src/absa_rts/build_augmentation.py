from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build size-matched targeted augmentation JSONL sets from ABSA-RTS pairs."
    )
    parser.add_argument("--pairs-csv", required=True, help="Path to absa_rts_pairs.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--size", type=int, default=8, help="Examples per single-category set")
    return parser.parse_args()


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_instruction(row: dict[str, str]) -> dict[str, str]:
    return {
        "instruction": "Classify sentiment for the given aspect from the review text. Output strict JSON: {\"aspect\": \"...\", \"sentiment\": \"positive|negative|neutral\"}.",
        "input": json.dumps({"text": row["followup_text"], "aspect": row["aspect"]}, ensure_ascii=False),
        "output": json.dumps(
            {"aspect": row["aspect"], "sentiment": row["expected_followup_sentiment"]},
            ensure_ascii=False,
        ),
        "category": row["category"],
        "relation_type": row["relation_type"],
        "pair_id": row["pair_id"],
    }


def sample_category(rows: list[dict[str, str]], category: str, k: int, rng: random.Random) -> list[dict[str, str]]:
    pool = [r for r in rows if r["category"] == category]
    if not pool:
        raise ValueError(f"No rows for category: {category}")
    if len(pool) >= k:
        return rng.sample(pool, k)
    out = []
    for i in range(k):
        out.append(pool[i % len(pool)])
    rng.shuffle(out)
    return out


def write_jsonl(path: Path, items: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    pairs = read_pairs(Path(args.pairs_csv))
    out_dir = Path(args.output_dir)

    c1_rows = sample_category(pairs, "negation", args.size, rng)
    c2_rows = sample_category(pairs, "contrast", args.size, rng)
    c4_rows = c1_rows + c2_rows

    c1 = [to_instruction(r) for r in c1_rows]
    c2 = [to_instruction(r) for r in c2_rows]
    c4 = [to_instruction(r) for r in c4_rows]

    write_jsonl(out_dir / "c1_negation_only.jsonl", c1)
    write_jsonl(out_dir / "c2_contrast_only.jsonl", c2)
    write_jsonl(out_dir / "c4_all_categories.jsonl", c4)

    manifest = {
        "seed": args.seed,
        "single_category_size": args.size,
        "files": {
            "c1_negation_only.jsonl": len(c1),
            "c2_contrast_only.jsonl": len(c2),
            "c4_all_categories.jsonl": len(c4),
        },
        "notes": "C1/C2 are size-matched by design. C4 is the union.",
    }
    (out_dir / "augmentation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote augmentation sets to {out_dir}")


if __name__ == "__main__":
    main()

