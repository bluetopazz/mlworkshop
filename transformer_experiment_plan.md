# Transformer Experiment Plan

## Highest-Value First Runs

1. Tokenizer training dominated by FineWeb Edu, with a small science/systems mix from the repo.
2. Dense base pretrain with RoPE + RMSNorm + SwiGLU + GQA on FineWeb Edu as the main corpus.
3. Dense domain-adapt continuation on FineWeb Edu plus repo-local science/systems/reasoning text.
4. Small SFT run on curated local reasoning/instruction examples.
5. MoE matched-active-compute comparison against the dense baseline.
6. Reward-guided refinement on math/code/reasoning style outcome tasks.

## Core Ablations

### Backbone

- absolute positions vs RoPE
- LayerNorm vs RMSNorm
- GELU MLP vs SwiGLU
- full MHA vs GQA
- QK norm off vs on

### Sparse path

- dense vs MoE
- top-1 vs top-2 routing
- MoE every 2 layers vs every 4 layers

### Data recipe

- FineWeb Edu dominated base mix vs FineWeb Edu plus heavier repo-local mix
- base-only vs base then domain-adapt
- no synthetic reasoning data vs synthetic reasoning augmentation
- tokenizer on broad-only vs tokenizer on broad-plus-specialization mix

### Post-training

- pretrain-only vs SFT
- SFT vs SFT + preference/reward refinement

## Metrics To Track

- train loss
- val loss
- perplexity
- domain perplexity
- repo-science perplexity
- prompt-level reasoning score
- code/math exact-match or verifier score
- tokens/sec
- memory use
- step time
- gradient norm
- parameter/update norm
- router entropy and expert utilization for MoE

## Notebook Flows To Run First

1. Setup, repo-aware local corpus recipe, and contamination/eval manifest cells.
2. FineWeb Edu loading/cache cell and combined phase text export cells.
3. Dense model config plus tiny smoke test.
4. Launch bundle generation for base pretraining.
5. Dashboard cells after a first metrics file exists.
6. Attention/residual/router probes on a checkpointed model.
7. SFT and reward-guided cells after the dense baseline is stable.
