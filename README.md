# mlworkshop

A research-oriented workshop repository for **scientific machine learning**, **physics-aware neural modeling**, **agentic model selection**, and **transformer systems experimentation**.

This repository collects four major project lines:

1. **Turbulence forecasting with physics-aware neural models**
2. **Magnetohydrodynamics (MHD) forecasting with hybrid PINN-style constraints**
3. **An agentic scientific modeling system for PDE/ODE expert selection**
4. **A small decoder-only transformer trained to study attention, scaling, and distributed training**

The unifying theme across all of them is the same:

> how should we design machine learning systems that respect structure—whether that structure comes from physics, numerical methods, conservation laws, sequence modeling, or model architecture itself?

---

# Table of contents

- [Repository vision](#repository-vision)
- [Project overview](#project-overview)
- [1. Turbulence forecasting](#1-turbulence-forecasting)
  - [Goal](#goal)
  - [Dataset](#dataset)
  - [Field schema](#field-schema)
  - [System architecture](#system-architecture)
  - [Training design](#training-design)
  - [Evaluation and diagnostics](#evaluation-and-diagnostics)
  - [Why this project matters](#why-this-project-matters)
- [2. MHD forecasting](#2-mhd-forecasting)
  - [Goal](#goal-1)
  - [Dataset](#dataset-1)
  - [Field schema](#field-schema-1)
  - [System architecture](#system-architecture-1)
  - [Training design](#training-design-1)
  - [Evaluation and diagnostics](#evaluation-and-diagnostics-1)
  - [Why this project matters](#why-this-project-matters-1)
- [3. Agentic scientific modeling system](#3-agentic-scientific-modeling-system)
  - [Goal](#goal-2)
  - [Core idea](#core-idea)
  - [System architecture](#system-architecture-2)
  - [Supported expert families](#supported-expert-families)
  - [Planning and belief update](#planning-and-belief-update)
  - [Training and scoring](#training-and-scoring)
  - [Why this project matters](#why-this-project-matters-2)
- [4. Small transformer / distributed training project](#4-small-transformer--distributed-training-project)
  - [Goal](#goal-3)
  - [Data sources](#data-sources)
  - [Tokenizer and corpus design](#tokenizer-and-corpus-design)
  - [Model architecture](#model-architecture)
  - [Training system](#training-system)
  - [Diagnostics and analysis](#diagnostics-and-analysis)
  - [Why this project matters](#why-this-project-matters-3)
- [Cross-cutting themes](#cross-cutting-themes)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Outputs and artifacts](#outputs-and-artifacts)
- [Limitations](#limitations)
- [Future directions](#future-directions)

---

# Repository vision

`mlworkshop` is not meant to be a single polished package with one entrypoint. It is a **research workshop**: a place to build, test, compare, and understand machine learning systems that interact with real structure.

The codebase emphasizes:

- **named physical variables rather than anonymous tensors**
- **architecture comparison rather than single-model attachment**
- **physics-aware diagnostics rather than only generic loss**
- **transparent experimentation**
- **notebook-first scientific development**
- **system-level understanding of training pipelines**

Across the repository, the main questions are:

- When does adding **physical inductive bias** help?
- How should we compare **classical** vs **hybrid physics-aware** models?
- Can an **agentic system** decide which scientific neural architecture is appropriate for a problem?
- What do we actually learn by building a transformer from scratch, including tokenization, distributed training, and diagnostics?

---

# Project overview

## Included projects

### A. Turbulence forecasting
A 3D physical field forecasting workflow for **turbulence with gravity and cooling**, based on **The Well** dataset. This project compares classical convolutional forecasting against more structured, hybrid physics-aware models.

### B. MHD forecasting
A 3D scientific forecasting pipeline for **compressible magnetohydrodynamics**, also based on **The Well**. This project emphasizes field structure, divergence penalties, induction-like constraints, and hybrid PINN-style regularization.

### C. Agentic scientific modeling
A scientific workflow agent that reasons over PDE/ODE problems, proposes modeling hypotheses, routes among expert families like **PINNs**, **FNOs**, **HNNs**, and **LNNs**, and uses critic-based acceptance criteria.

### D. Small transformer / distributed training
A decoder-only language model project built to understand **attention**, **tokenization**, **training dynamics**, and **distributed training**. It includes corpus building, SentencePiece tokenization, PyTorch/FSDP training, and extensive diagnostics.

---

# 1. Turbulence forecasting

## Goal

The turbulence project is designed to answer a simple but important question:

> Can we build neural forecasting systems for complex 3D physical fields that remain scientifically interpretable and preserve important structures better than generic tensor-based training?

Rather than treating the data as an unlabeled multichannel volume, the project explicitly names and tracks physical fields. It aims to be a **scientific ML workflow**, not just a generic forecasting benchmark.

---

## Dataset

### Source
This project uses **The Well** dataset, specifically the:

- `turbulence_gravity_cooling` environment

### Metadata assumptions
The dataset is treated as:

- 3D
- spatial resolution: `64 x 64 x 64`
- open boundary conditions
- physically meaningful channels

### Raw fields used

Scalar fields:
- density
- pressure
- temperature

Vector fields:
- velocity_x
- velocity_y
- velocity_z

### Data loading path
The workflow uses the official WellDataModule / Hugging Face data path:

- base: `hf://datasets/polymathic-ai/`
- dataset: `turbulence_gravity_cooling`

### Caching strategy
Because streaming high-dimensional data repeatedly is expensive, the notebook builds persistent caches for:

- first canonical batch
- first canonical sample
- a canonical persisted mini-dataset
- streamed quartets for source-style trajectory visualizations

This lets experimentation continue across notebook restarts.

---

## Field schema

One major design decision in this project is replacing a generic “6-channel tensor” view with a physically named field interface:

- `0`: density
- `1`: pressure
- `2`: temperature
- `3`: velocity_x
- `4`: velocity_y
- `5`: velocity_z

This matters because almost every later component depends on field semantics:

- visualization
- losses
- diagnostics
- agent context packs
- physical interpretation

The project also derives:

- `velocity_magnitude`

to evaluate flow structure beyond componentwise error.

---

## System architecture

The turbulence workflow has multiple layers.

### 1. Canonical data interface
The raw The Well batch format is canonicalized into:

- input tensor `x`: `[B, C, D, H, W]`
- target tensor `y`: `[B, C, D, H, W]`
- constant scalars
- time grids
- spatial grid
- metadata

### 2. Visualization layer
The project includes:

- named field pair plots
- trajectory snapshots
- multi-field summary grids
- source-style slice visualizations

These are meant to make the workflow interpretable to a human scientist.

### 3. Context-pack layer
A structured “agent context pack” summarizes:
- task type
- spatial resolution
- boundary conditions
- constant scalars
- input field statistics
- time-step information
- notes about the physical regime

This is meant to bridge raw simulation data and higher-level orchestration systems.

### 4. Forecasting models

#### Classical baseline
A lightweight 3D residual CNN forecaster:
- Conv3D projection in
- several residual blocks
- Conv3D projection out
- predicts next-state residuals

Form:
- input: `[B, 6, D, H, W]`
- output: `[B, 6, D, H, W]`
- residual prediction: `y_hat = x + f(x)`

#### Composite hybrid model
A more structured hybrid model decomposes the prediction into:

- a **flow / thermodynamic backbone**
- a **pressure closure head**

The factorization is:

- `z_(t+dt) = H_phi(x_t, c)`
- `p_(t+dt) = Pi_psi(x_t, c, z_(t+dt))`

where:
- `z = [density, temperature, velocity_x, velocity_y, velocity_z]`
- pressure is modeled separately afterward

This staged formulation is used to reduce gradient interference between tasks.

---

## Training design

### Classical model loss
The turbulence classical forecaster is trained with a composite physics-aware objective that includes:

1. global data MSE
2. density-channel MSE
3. dense-region weighted density loss
4. density-gradient consistency
5. velocity-magnitude consistency
6. mass-consistency penalty

These choices reflect the idea that not all physical errors are equally important. Dense regions and structural features matter.

### Composite hybrid training phases
The staged hybrid model is trained in three phases:

#### Phase A: flow pretraining
Train density, temperature, and velocity first.

#### Phase B: pressure fitting
Freeze the flow backbone and train the pressure head.

#### Phase C: joint fine-tuning
Jointly optimize everything with a controlled schedule so pressure does not destabilize the trunk.

### Physics-inspired regularization
The hybrid model includes reduced physical residuals such as:

- continuity-like residuals
- momentum-style residuals
- light EOS-style pressure consistency

The goal is not to build a full numerical solver, but to encourage physically coherent behavior.

---

## Evaluation and diagnostics

This project avoids relying only on global MSE.

### Quantitative metrics
Examples include:
- global MSE
- channelwise MSE
- relative L2 by field
- velocity magnitude relative L2

### Structure-aware diagnostics
These include:
- dense-region IoU
- density-gradient comparison
- dense-region overlap visualization
- velocity magnitude error maps

### Qualitative diagnostics
Plots compare:
- input
- target
- prediction
- absolute error

for selected fields and slices.

### Reporting
A JSON report is saved with:
- metadata
- example context pack
- evaluation metrics
- dense-region audit
- cache information
- timestamp and device information

---

## Why this project matters

This project is an attempt to build scientific neural forecasting in a way that stays close to the structure of the underlying system.

The most important ideas are:
- physically named fields
- architecture comparison
- structure-aware losses
- staged factorization for coupled outputs
- field-specific diagnostics instead of pure aggregate loss

It is relevant to scientific ML because it asks how we should train neural models on PDE-governed systems without losing the meaning of the variables.

---

# 2. MHD forecasting

## Goal

The MHD project asks a related but more demanding question:

> Can we forecast 3D compressible magnetohydrodynamic states with models that respect both field structure and reduced physical constraints?

Compared with turbulence alone, MHD introduces:
- magnetic fields
- divergence-free structure
- induction-like dynamics
- stronger vector-field coupling

This makes it a useful testbed for hybrid scientific ML.

---

## Dataset

### Source
This project uses **The Well** dataset:

- `MHD_64`

### Metadata assumptions
The dataset is treated as:
- 3D
- Cartesian
- spatial resolution `64 x 64 x 64`
- periodic boundary conditions

### Constant scalars
The workflow tracks:
- `Ma`: Alfvénic Mach number
- `Ms`: sonic Mach number

### Data loading
Again, the workflow uses The Well’s official Hugging Face / datamodule loading path.

### Persistent caching
As with the turbulence workflow, the project caches:
- first batch
- first sample
- a canonical mini-dataset
- quartet trajectories for repeated field visualization

---

## Field schema

The MHD project uses a 7-channel field schema:

- `0`: density
- `1`: magnetic_field_x
- `2`: magnetic_field_y
- `3`: magnetic_field_z
- `4`: velocity_x
- `5`: velocity_y
- `6`: velocity_z

Derived quantities:
- magnetic magnitude
- velocity magnitude

This lets the system evaluate both raw components and more physically meaningful derived fields.

---

## System architecture

### 1. Canonical scientific data interface
The raw streamed MHD data is canonicalized into:
- input state `x`
- target state `y`
- constants
- grid
- time information
- metadata

### 2. Visualization layer
The project includes:
- named field pair plots
- trajectory snapshots
- summary grids across density / magnetic / velocity structure
- raw-file inspection where available

### 3. Agent context pack
The MHD context pack summarizes:
- task type
- field schema
- boundary conditions
- Mach-number scalars
- input statistics
- notes about physical regime

### 4. Forecasting models

#### Classical forecaster
A residual 3D CNN:
- input: `[B, 7, D, H, W]`
- output: `[B, 7, D, H, W]`
- residual next-state prediction

#### Hybrid MHD PINN-style model
A conditioning-aware residual neural model:
- input state `x`
- constant scalars `[Ma, Ms]`
- output next-state `y_hat`

This hybrid model is trained with additional physics-aware penalties inspired by reduced MHD structure.

---

## Training design

### Classical loss
The classical MHD model uses a composite loss with terms for:

1. global MSE
2. density MSE
3. density-structure region loss
4. density-gradient consistency
5. magnetic magnitude preservation
6. velocity magnitude preservation
7. divergence(B) penalty
8. continuity-like residual

### Hybrid loss
The hybrid MHD PINN-style loss extends the classical objective with:

- all core data-fit terms
- stronger divergence-free magnetic penalties
- continuity-like residual
- induction-like residual from `dB/dt - curl(v x B)`

Again, the idea is not to reconstruct a full exact MHD solver, but to make the learned dynamics less physically arbitrary.

---

## Evaluation and diagnostics

### Quantitative metrics
- global MSE
- fieldwise MSE
- fieldwise relative L2
- magnetic magnitude relative L2
- velocity magnitude relative L2

### Physics-aware audits
- structure-region IoU on density
- mean absolute divergence of predicted magnetic field
- structure-overlap comparison between classical and hybrid models

### Qualitative plots
For selected slices:
- target vs classical prediction
- target vs hybrid prediction
- density structure overlap
- magnetic magnitude error
- velocity magnitude error

### Reporting
The project saves a report block with:
- metadata
- context pack
- classical evaluation
- hybrid evaluation
- structure/divB audits
- model checkpoint locations

---

## Why this project matters

MHD is a useful proving ground because it forces the modeling workflow to handle:

- vector-valued physical fields
- derived invariant-like structure
- divergence constraints
- coupled fluid and magnetic behavior

The project matters as scientific ML because it explores the practical middle ground between:
- purely data-driven neural forecasting
- expensive or rigid fully physics-imposed modeling

It is a study in **reduced physical structure as inductive bias**.

---

# 3. Agentic scientific modeling system

## Goal

This project asks a different question:

> Can an agentic system reason about a scientific modeling problem, choose among candidate neural expert families, and evaluate them using structure-aware criteria?

This is not an LLM pretending to solve physics directly.

Instead, the design separates:
- hypothesis planning
- architecture selection
- numerical fitting
- criticism / acceptance
- memory and reporting

The goal is to create a workflow where language-guided reasoning supports scientific modeling without replacing it.

---

## Core idea

The system is built around the following loop:

1. summarize current evidence
2. build a bounded belief and uncertainty state
3. propose or refine a hypothesis
4. route to scientific expert families
5. fit candidate experts
6. compare them with scientific diagnostics
7. accept / reject based on critic logic
8. update memory and continue if needed

The system is meant to be:
- explicit
- inspectable
- schema-constrained
- modular

---

## System architecture

### 1. Research state
The central state object stores:
- domain
- data fingerprint
- current hypothesis
- candidate hypotheses
- session memory
- evidence graph
- belief
- uncertainty
- accepted/rejected runs
- best metrics
- action history
- notes
- final report

### 2. Synthetic data generators
The project includes synthetic scientific environments for:
- Burgers-like PDE systems
- pendulum-like ODE systems

These are used to test the agent’s reasoning and routing behavior.

### 3. Expert registry
A registry maps expert-family names to:
- builders
- fit functions
- score functions

### 4. Planner
A schema-constrained planner proposes:
- hypothesis ID
- parameters / priors
- probes
- audits

The planner is bounded. It is not allowed to invent arbitrary action formats.

### 5. Belief update layer
A deterministic bounded reasoning layer summarizes evidence and updates:
- progress status
- uncertainty
- current bottleneck
- recommended next action

### 6. Critic
A critic decides whether a new run improves enough to be accepted based on domain-specific metrics.

### 7. Session and evidence stores
The system logs:
- session summaries
- evidence graph events
- accepted and rejected runs
- agent action history

This keeps the workflow auditable.

---

## Supported expert families

The registry supports multiple scientific model families.

### For PDE-style systems
- **SA-PINN**
- **DeepHPMs**
- **FNO-lite**

### For ODE-style systems
- **HNN**
- **LNN**

Each family is wrapped with:
- build
- fit
- score

interfaces so the agent can compare them consistently.

---

## Planning and belief update

### Hypothesis space
The planner supports hypotheses such as:
- PDE hyperbolic / parabolic style assumptions
- ODE conservative vs generic structure

### Proposal schema
A valid proposal includes:
- `hypothesis_id`
- `params`
- `probes`
- `audits`

### Belief state
The belief update layer interprets evidence into categories like:
- no proposal
- no baseline
- good enough
- plateaued
- unsatisfactory

It also identifies bottlenecks such as:
- poor fit
- long-horizon drift
- family mismatch
- physics residual issues

This bounded reasoning is what makes the agent “agentic” without making it vague.

---

## Training and scoring

### PDE scoring
PDE experts are scored on quantities like:
- data MSE
- physics residual
- finite-difference residual audit
- relative L2 error

### ODE scoring
ODE experts are scored on:
- vector-field fit
- short-horizon trajectory error
- short-horizon conservation drift
- long-horizon drift

### Tournament logic
The agent runs candidate experts, scores them, normalizes metric values, and picks a winner by aggregate score.

For ODE problems it can also separately identify:
- best local vector-field expert
- best conservative expert

### Acceptance logic
The critic checks whether new runs improve enough relative to the current best system, using:
- relative improvements
- drift ceilings
- satisfactory thresholds

This makes the loop closer to actual research iteration than to a one-shot benchmark.

---

## Why this project matters

This project matters because scientific ML often involves **model family choice**, not only parameter tuning.

It explores a useful meta-question:

> how do we select among inductive biases for a scientific system?

The project is especially relevant at the boundary of:
- scientific machine learning
- model-based reasoning
- agentic systems
- automated experimentation

Its main contribution is architectural:
- separate planner from solver
- keep reasoning bounded
- use scientific critics instead of generic scores alone

---

# 4. Small transformer / distributed training project

## Goal

This project was built to understand transformer systems from the inside:

- attention
- tokenization
- corpus construction
- scaling constraints
- distributed training
- hidden-state geometry
- FSDP-based training

The goal is not just “train a language model,” but:

> build enough of the pipeline directly to understand what each subsystem is doing.

---

## Data sources

### Local / curated documents
The notebook includes a small local curated document store to inject domain-specific content and custom writing.

### Web corpus
The broader pretraining data is built from streamed web corpora, especially:

- **FineWeb Edu** (`fineweb_edu`)

The workflow is designed so that the model is not trained only on tiny pasted text. It uses a much larger background corpus.

### Phase design
The corpus is split into phases:

- tokenizer phase
- base pretraining phase
- adaptation phase

This explicitly separates broad language modeling from later domain shaping.

---

## Tokenizer and corpus design

### Tokenizer
The project trains a **SentencePiece** tokenizer with:
- unigram model
- vocabulary size around `16k`

### Corpus preparation
The workflow:
- streams web documents
- cleans and segments local documents
- chunks long web documents
- removes exact duplicates
- builds weighted phase corpora

### Token streams
Encoded corpora are exported to flat token streams on disk:
- base train / val tokens
- adapt train / val tokens

This design avoids giant in-memory segment objects and makes later training more stable.

---

## Model architecture

The model is a decoder-only transformer with:

- token embeddings
- positional embeddings
- causal self-attention blocks
- MLP blocks
- layer normalization
- tied output weights

A target scale is approximately **500M-class**, though this project is best understood as a systems-learning exercise in implementing and training a nontrivial transformer.

### Architecture details
Examples from the config include:
- `d_model`
- `n_heads`
- `n_layers`
- `mlp_ratio`
- dropout
- tied embeddings

The model module also stores internal diagnostics such as:
- attention scores
- attention probabilities
- Q/K/V norms
- entropy by head
- hidden-state outputs by layer

---

## Training system

### Distributed training
The project uses:
- **PyTorch**
- **FSDP**
- multi-GPU training
- bf16 mixed precision where available
- activation checkpointing
- checkpoint save/load
- resume support

### Worker design
Training is separated into:
- notebook orchestration
- a distributed worker script

This allows:
- launch / monitoring from notebook
- resilient checkpointing
- live dashboards without embedding all training logic in notebook cells

### Optimization
The training stack includes:
- AdamW
- learning-rate warmup
- cosine decay
- gradient clipping
- gradient accumulation

### Training phases
The system supports:
- base training
- adapt training
- resume from latest checkpoint
- initialize adapt from base checkpoint

---

## Diagnostics and analysis

This is one of the main strengths of the project.

### Notebook dashboards
The project includes dashboards for:
- train / validation loss
- grad norm
- learning rate
- tokens/sec
- rank-0 memory usage

### Attention diagnostics
The model can visualize:
- attention probabilities
- raw attention scores
- mean attention received by source position
- entropy per target position
- same head across layers

### Representation diagnostics
It also visualizes:
- token embeddings in 3D PCA space
- hidden-state geometry across layers
- layerwise residual norms
- Q/K/V norm statistics
- head diversity statistics

### Next-token probability dashboard
The notebook can inspect:
- top-k next-token probabilities
- cumulative mass
- entropy of the predictive distribution

### KV-cache accounting
There is also a conceptual memory-analysis notebook component for KV-cache scaling.

---

## Why this project matters

This project matters less because of the raw fact of training a transformer, and more because it demonstrates system-level understanding of:

- corpus design
- tokenizer design
- sequence-model training
- distributed training mechanics
- attention diagnostics
- representational analysis

It complements the scientific ML projects by showing a second axis of interest:
- neural architecture understanding
- scaling systems
- modern training pipelines

---

# Cross-cutting themes

Although the repository contains multiple different projects, they share a few core themes.

## 1. Named structure over generic tensors
The physics projects explicitly name fields.
The agent project explicitly names hypothesis spaces and expert families.
The transformer project explicitly names phases, diagnostics, and token pipelines.

## 2. Scientific evaluation over single scalar loss
Across the repository, evaluation is richer than one metric:
- structure overlap
- divergence audits
- conservation drift
- fieldwise relative error
- critic acceptance logic
- hidden-state and attention diagnostics

## 3. Architecture comparison
The repository is not organized around one model ideology.
Instead, it repeatedly asks:
- what kind of model is appropriate here?
- what inductive bias is helping?
- what are the tradeoffs?

## 4. Notebook-first research development
These projects are developed as research notebooks and experimental workflows, prioritizing:
- interpretability
- debuggability
- live plotting
- cached artifacts
- transparent iteration

## 5. Systems understanding
Even in the transformer project, the point is not only final generations.
The point is understanding:
- how the training system works
- how memory scales
- how attention behaves
- how representations evolve

---
