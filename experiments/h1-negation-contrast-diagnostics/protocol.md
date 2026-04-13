# Protocol: H1 Negation/Contrast Diagnostics

## Hypothesis

H1: Prompt-only (A0) and standard QLoRA SFT (A1) models show larger robustness drops on negation/contrast transformations than on non-transformed ABSA ASC items.

## Why This Test

Negation and contrast are high-frequency linguistic operators in proposal-defined failure modes and should produce measurable reliability gaps if robustness is brittle.

## Confirmatory Setup

1. Build deterministic metamorphic templates for:
   - negation insertion/removal with expected polarity flip
   - contrast with target-aspect sentiment anchored in post-contrast clause
2. Construct ABSA-RTS subset (Restaurants first):
   - 50 negation examples
   - 50 contrast examples
3. Evaluate A0 and A1 on:
   - original items
   - transformed items
4. Metrics:
   - metamorphic pass rate per category
   - category accuracy
   - macro-F1 on standard SemEval test split (guardrail)

## Success / Failure Criteria

- **Supports H1:** measurable and repeatable pass-rate drop on transformed negation/contrast relative to original slice.
- **Refutes H1:** no substantial drop beyond expected noise.

## Data and Leakage Guard

ABSA-RTS diagnostics must remain evaluation-only and separated from training data used in A1 and later A2/C1-C4 runs.

## Execution Notes

- Domain order: Restaurants first, then Laptops after pipeline stability.
- Ambiguous transformed cases must be removed before scoring.

## Colab Execution

Run the notebook:

- `experiments/h1-negation-contrast-diagnostics/code/h1_colab_runner.ipynb`

It includes:
1. Project path setup for Colab.
2. ABSA-RTS generation.
3. A1 prediction-file generation (placeholder).
4. A0/A1 evaluation and output inspection.
