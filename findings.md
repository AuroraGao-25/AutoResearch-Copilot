# Research Findings

## Research Question

How can we make instruction-tuned ABSA models robust to known linguistic failure modes (especially negation, contrast, and multi-aspect conflict) beyond what aggregate macro-F1 captures?

## Current Understanding

The proposal and early literature converge on the same gap: high benchmark scores can coexist with predictable, category-specific failures. The strongest path is to treat ABSA robustness as a software testing problem, where deterministic metamorphic relations and category-stratified diagnostics become first-class evaluation targets, then link targeted augmentation directly to those targets through ablations.

## Key Results

- Bootstrap complete: research state, literature corpus, and first experiment protocol are now in place.
- High-priority hypothesis selected: H1 (negation/contrast vulnerability under A0/A1).
- First experiment scaffold created under `experiments/h1-negation-contrast-diagnostics/`.
- Execution pipeline implemented in `src/absa_rts/` with Colab notebook runner for direct cloud execution.
- H1 v2 results from Colab: A0 metamorphic pass rate = 1.0, A1 metamorphic pass rate = 1.0 on current synthetic set.

## Patterns and Insights

- CheckList-style behavioral evaluation is directly compatible with ABSA-RTS design.
- Proposal-defined categories naturally map to testable metamorphic relations with expected invariance/flip behavior.
- The contribution boundary is clearest when each augmentation category has a matching robustness target and ablation.
- Current synthetic templates are too easy for the proxy evaluator and cannot reveal robustness boundary differences.

## Lessons and Constraints

- Semantic Scholar API access is currently rate-limited in this environment; arXiv and CrossRef are reliable backups for bootstrap.
- Robustness claims must remain deterministic-by-construction to avoid ambiguous metamorphic labels.
- Standard SemEval performance must be tracked as a guardrail, not the optimization objective.
- Negation templates must preserve semantic direction (`not good` => negative, `not bad` => positive); prior inversion can create artificial zero pass-rate.

## Open Questions

- Which exact template families for negation and contrast maximize diagnostic signal while preserving label determinism?
- How strongly will metamorphic pass rate correlate with macro-F1 by domain and category?
- What tolerance margin on SemEval macro-F1 should define "no harmful regression" for A2 and C1-C4?
- Will C1/C2 targeted augmentation show category-specific gains when evaluated with real model outputs instead of heuristic proxy predictions?

## Optimization Trajectory

No model runs yet. Trajectory starts after baseline A0/A1 metrics are recorded.
