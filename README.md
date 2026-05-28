<div align="center">

# Do Recurrent RL Agents Discover a Unique Internal Clock?

**Evidence Against a Canonical Temporal Code**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/temporal_repr.pdf)
[![License](https://img.shields.io/badge/License-MIT-green)]()

</div>

---

**English** | [中文](#中文版)

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
<img src="paper/figures/fig3_horizon_sensitivity.png" width="500">
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
- **Venue**: Finding the Frame Workshop @ RLC 2026
- **PDF**: [`paper/temporal_repr.pdf`](paper/temporal_repr.pdf)

## Citation

```bibtex
@inproceedings{temporal_repr_2026,
  title={Do Recurrent RL Agents Discover a Unique Internal Clock? Evidence Against a Canonical Temporal Code},
  author={Anonymous},
  booktitle={Finding the Frame Workshop at Reinforcement Learning Conference},
  year={2026}
}
```

---

<a id="中文版"></a>

# 循环 RL 智能体会发现唯一的内部时钟吗？

**反对规范时间编码的证据**

<div align="right"><a href="#top">English</a></div>

## 一句话总结

我们在周期性任务上训练 LSTM 智能体，发现**不同的随机种子产生根本不同的时间内部表征**——尽管行为表现完全相同。不存在单一的"规范"时间编码。

## 核心发现

| | Seed 42 | Seed 456 |
|---|---|---|
| **Return** | 0.920 | 0.920 |
| **R²(时间步)** | 1.000 | 1.000 |
| **最匹配的编码** | 正弦 PE (0.74) | 相位 sin/cos (0.62) |

相同的行为，不同的内部时钟。

## 时间范围效应

随着时间范围增加，智能体的表征从正弦编码转向标量编码：

<p align="center">
<img src="paper/figures/fig3_horizon_sensitivity.png" width="500">
</p>

## 快速开始

```bash
conda activate rlenv
python run_all.py --phase all       # 一键复现
python run_all.py --phase train     # 仅训练
python run_all.py --phase analyze   # 仅分析
```

## 引用

```bibtex
@inproceedings{temporal_repr_2026,
  title={Do Recurrent RL Agents Discover a Unique Internal Clock? Evidence Against a Canonical Temporal Code},
  author={Anonymous},
  booktitle={Finding the Frame Workshop at Reinforcement Learning Conference},
  year={2026}
}
```
