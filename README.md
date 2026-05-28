<div align="center">

# Do Recurrent RL Agents Discover a Unique Internal Clock?

**Evidence Against a Canonical Temporal Code**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/temporal_repr.pdf)
[![License](https://img.shields.io/badge/License-MIT-green)]()
[**中文**](README_CN.md)

</div>

---

## TL;DR

We train LSTM agents on a periodic task and find that **different random seeds produce fundamentally different internal representations of time** — despite identical behavioral performance. There is no single "canonical" temporal code.

## Key Finding

| | Seed 42 | Seed 456 |
|---|---|---|
| **Return** | 0.920 | 0.920 |
| **R²(timestep)** | 1.000 | 1.000 |
| **Best CKA match** | Sinusoidal PE (0.74) | Phase sin/cos (0.62) |

Same behavior, different internal clocks.

## Horizon Effect

As temporal horizon increases, the agent's representation shifts from sinusoidal-like to scalar-like encoding:

<p align="center">
<img src="paper/figures/fig3_horizon_sensitivity.png" width="480">
</p>

## Repository Structure

```
├── paper/              # Paper (PDF, LaTeX, BibTeX)
├── agents/             # LSTM & MLP actor-critic
├── environments/       # DoorGrid + temporal wrappers
├── analysis/           # CKA, linear probe, PCA, MI
├── experiments/        # Training & evaluation scripts
├── figures/            # Per-seed result JSONs
├── results/models/     # Trained model weights (.pt)
├── run_all.py          # Reproduce everything
└── compute_cross_seed_cka.py
```

## Quick Start

```bash
conda activate rlenv
python run_all.py --phase all
```

Or step by step:

```bash
python run_all.py --phase train     # Train all agents
python run_all.py --phase analyze   # Run analysis
python run_all.py --phase figures   # Generate figures
```

## Paper

- **Title**: Do Recurrent RL Agents Discover a Unique Internal Clock? Evidence Against a Canonical Temporal Code
- **PDF**: [`paper/temporal_repr.pdf`](paper/temporal_repr.pdf)

## Citation

```bibtex
@article{temporal_repr_2026,
  title={Do Recurrent RL Agents Discover a Unique Internal Clock? Evidence Against a Canonical Temporal Code},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```
