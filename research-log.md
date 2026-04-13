# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-04-13 | bootstrap | Initialized autoresearch workspace from proposal.md. Defined H1-H3, proxy metrics, and first experiment track focused on negation/contrast diagnostics in ABSA-RTS. |
| 2 | 2026-04-13 | bootstrap | Literature bootstrap from CrossRef + arXiv completed. Collected core benchmark, behavioral testing, and PEFT papers into literature/ with structured summaries. |
| 3 | 2026-04-13 | outer-loop | Direction DEEPEN: prioritize high-signal failure categories (negation, contrast) and lock deterministic metamorphic protocol before model runs. |
| 4 | 2026-04-13 | inner-loop | Implemented ABSA-RTS construction and baseline evaluation scripts in `src/absa_rts/` plus a Colab notebook runner at `experiments/h1-negation-contrast-diagnostics/code/h1_colab_runner.ipynb` for GPU-based execution. |
| 5 | 2026-04-13 | inner-loop | Corrected negation metamorphic transformation semantics (`not good` / `not bad`) and added `h1_colab_runner_v2.ipynb` with a git-sync step for rerunnable Colab execution. |
