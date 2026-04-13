# Protocol: H2 Targeted Augmentation (C1/C2/C4)

## Hypothesis

H2: Single-category targeted augmentation improves its matching category metric more than non-target categories, while C4 provides broader but less concentrated gains.

## Why this follows H1

H1 v2 currently shows saturated metamorphic pass rate (1.0), so we need a stronger next step: build controlled augmentation sets first, then plug them into real A1/A2 model training where effects are measurable.

## Confirmatory Setup

1. Build size-matched instruction JSONL sets from ABSA-RTS pairs:
   - C1: negation only
   - C2: contrast only
   - C4: all categories (union)
2. Keep deterministic output format:
   - `{"aspect":"...","sentiment":"..."}`
3. Train/evaluate on Colab with your real model:
   - A1 baseline
   - A2+C1
   - A2+C2
   - A2+C4
4. Compare:
   - per-category metamorphic pass rate
   - macro-F1 guardrail

## Current Deliverable in Repo

- Data-prep script: `src/absa_rts/build_augmentation.py`
- Colab prep notebook: `experiments/h2-targeted-augmentation/code/h2_colab_prep.ipynb`

