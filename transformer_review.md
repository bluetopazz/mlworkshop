# Transformer Stack Review

## What Exists

The repository is small and coherent. The scientific notebooks are strongly aligned with the workshop's identity:

- `turbulence.ipynb` keeps named physical fields, structure-aware losses, and scientist-readable diagnostics.
- `MHD_64.ipynb` extends that style with stronger physical numerics, staged hybrid training, and richer audits.
- `scienceagentv1.ipynb` is a bounded, inspectable orchestration notebook rather than an opaque agent demo.
- `transformer.ipynb` already goes beyond a toy LM: it has phase-aware corpora, SentencePiece tokenization, multi-GPU FSDP training, checkpoint/resume logic, and analysis dashboards.

The transformer's current strengths are:

- explicit base/adapt phase split
- disk-backed token export
- distributed training separation via a worker script
- activation checkpointing and bf16 support
- notebook-native diagnostics for attention, hidden-state geometry, and next-token uncertainty

## What Is Weak Or Broken

The current transformer stack is promising, but it is still much closer to a strong learning exercise than to a modern small frontier-style recipe.

### Data and tokenization

- `RAW_DOCUMENTS` is empty, so the curated local/domain signal is effectively absent.
- Local-corpus construction is manual rather than repo-aware; the notebook does not leverage `README.md`, `agents.md`, or notebook markdown as structured local data.
- Deduplication is exact-hash only; there is no near-duplicate or contamination audit.
- The train/val split is positional over one flat stream, with no domain-slice eval construction.
- There is no explicit reasoning/math/code/science eval pack.
- There is no teacher-assisted or synthetic reasoning data path.

### Architecture

- The model is still a classic GPT-2 style stack: learned absolute position embeddings, LayerNorm, GELU MLP, full MHA only.
- There is no RoPE, RMSNorm, SwiGLU, or grouped-query attention.
- There is no cache-aware inference implementation in the model itself.
- There is no option for sliding-window attention, QK normalization, or sparse MoE blocks.
- Diagnostics are useful but mostly post hoc; they are not integrated with a richer model instrumentation surface.

### Training system

- The worker is emitted from the notebook instead of living as a reusable source file.
- Metrics are too narrow for serious systems work: no parameter norm, update norm, step-time breakdown, load-balancing stats, or instability alerts.
- There is no explicit stage machinery for SFT or reward-guided refinement.
- The notebook mixes experiment design, file generation, and launch logic more than it needs to.

### Dashboarding

- The current dashboards are useful but mostly Matplotlib line plots.
- There is no comparison dashboard across dense vs MoE, pretrain vs post-train, or old vs new recipes.
- There is no router dashboard, calibration view, checkpoint timeline, or throughput breakdown.
- The plots do not yet feel like a systems-research cockpit.

## What Should Remain Unchanged

These parts of the current design are worth preserving:

- notebook-first orchestration
- explicit phase separation
- disk-backed artifact handling
- clear training/eval checkpoints
- strong inspection affordances for attention and representations
- the overall repo narrative that scientific ML and transformer systems sit side by side

The scientific notebooks should remain mostly unchanged. They already express the project's identity well.

## What Must Be Refactored Immediately

1. Move transformer helper logic out of giant notebook string literals into importable local modules.
2. Replace the backbone with a modern dense baseline:
   - pre-norm
   - RMSNorm
   - RoPE
   - SwiGLU
   - grouped-query attention
   - optional QK normalization
   - cache-aware generation
3. Add an optional sparse MoE path without removing the dense baseline.
4. Add staged post-training support:
   - SFT-ready packing
   - preference/reward utilities
   - lightweight verifier/outcome scoring
5. Upgrade dashboards to interactive Plotly-based research views with instability and systems metrics.
6. Make the local corpus recipe repo-aware so the transformer's curated data is not effectively empty.

## What Should Be Deferred

These ideas are good, but they should follow after the dense baseline is stable:

- sliding-window/local-global attention as a measured efficiency experiment
- more sophisticated near-duplicate filtering
- a full distributed post-training worker path
- larger-scale synthetic data generation tied to an external teacher model
- richer benchmark harnesses with external datasets

## Gap Versus A Modern Small Frontier-Style Recipe

Compared with a modern strong small-model recipe, the current notebook is missing several high-leverage ingredients:

- stronger backbone defaults
- better quality/curation of local data
- curriculum-aware stage transitions
- richer eval slices
- optional sparse capacity path
- post-training and verifier-style improvement loop
- better systems instrumentation

The main difference is not just architecture. It is that modern strong small models win through the combination of:

- good data
- modern attention/MLP design
- stable training
- explicit stage transitions
- strong diagnostics

That combination is what this refactor should target.

## Immediate Research Direction

The best next-step stack for this repo is:

1. strong dense decoder-only baseline
2. optional sparse MoE variant on the same scaffold
3. curated SFT and preference/reward utilities
4. interactive dashboard layer
5. ablation-driven iteration rather than architecture ideology
