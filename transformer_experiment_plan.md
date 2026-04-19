# Transformer Experiment Plan

## Highest-Value First Runs

1. Dense base pretrain with RoPE + RMSNorm + SwiGLU + GQA.
2. Dense domain-adapt continuation on repo-local science/systems text.
3. Small SFT run on curated reasoning/instruction examples.
4. MoE matched-active-compute comparison against the dense baseline.
5. Reward-guided refinement on math/code/reasoning style outcome tasks.

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

- broad-only base mix vs broad + repo-local mix
- base-only vs base then domain-adapt
- no synthetic reasoning data vs synthetic reasoning augmentation

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
2. Dense model config plus tiny smoke test.
3. Launch bundle generation for base pretraining.
4. Dashboard cells after a first metrics file exists.
5. Attention/residual/router probes on a checkpointed model.
6. SFT and reward-guided cells after the dense baseline is stable.
