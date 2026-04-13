# ABSA Robustness Literature Survey

Date: 2026-04-13

## Focus

Robustness-oriented evaluation and targeted instruction tuning for Aspect Sentiment Classification (ASC) on SemEval domains.

## Source Coverage

- CrossRef: benchmark and evaluation venue metadata (SemEval series, CheckList).
- arXiv API: recent ABSA + instruction tuning + PEFT references.
- Semantic Scholar: attempted, but API rate-limited in current session (fallback used).

## Core Papers

1. **SemEval-2014 Task 4: Aspect Based Sentiment Analysis** (Pontiki et al., 2014)  
   Benchmark foundation for ABSA task definitions and splits.  
   URL: https://doi.org/10.3115/v1/s14-2004

2. **SemEval-2015 Task 12: Aspect Based Sentiment Analysis** (Pontiki et al., 2015)  
   Extension benchmark used for cross-year robustness context.  
   URL: https://doi.org/10.18653/v1/s15-2082

3. **SemEval-2016 Task 5: Aspect Based Sentiment Analysis** (Pontiki et al., 2016)  
   Expanded benchmark landscape and evaluation practice.  
   URL: https://doi.org/10.18653/v1/s16-1002

4. **Beyond Accuracy: Behavioral Testing of NLP Models with CheckList** (Ribeiro et al., 2021)  
   Direct methodological anchor for diagnostic suite + targeted bug pattern testing.  
   URL: https://doi.org/10.24963/ijcai.2021/659

5. **InstructABSA** (Scaria et al., 2023)  
   Instruction tuning strategy and data design for ABSA subtasks.  
   URL: https://arxiv.org/abs/2302.08624

6. **Large language models for ABSA** (Simmering and Huoviala, 2023)  
   Prompting/fine-tuning trade-offs and error behavior.  
   URL: https://arxiv.org/abs/2310.18025

7. **LoRA** (Hu et al., 2021) and **QLoRA** (Dettmers et al., 2023)  
   Practical low-memory fine-tuning basis for 7B model experiments in constrained GPU settings.  
   URLs: https://arxiv.org/abs/2106.09685, https://arxiv.org/abs/2305.14314

## Observed Gap (supports proposal)

Existing ABSA work often reports aggregate task metrics; less work provides deterministic metamorphic test suites that expose category-specific failures and ties targeted augmentation to those categories via controlled ablations.

## Immediate Research Direction

Start with H1: quantify failure concentration on negation and contrast using deterministic metamorphic tests and category-labeled ABSA-RTS slices.
