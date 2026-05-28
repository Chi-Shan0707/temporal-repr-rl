"""
experiments/run_high_roi.py — High-ROI improvements for V2 paper.

Improvements (in ROI order):
  1. Bootstrap CI for CKA on existing DoorGrid seeds (no training)
  2. T=8 DoorGrid multi-seed (4 new seeds)
  3. T={16,32} DoorGrid scaling experiments (5 seeds each)
  4. Collect task return data from all existing experiments
  5. Single-neuron tuning curve analysis

Usage:
  python run_high_roi.py --bootstrap          # Improvement 1 only
  python run_high_roi.py --t8-multiseed       # Improvement 2
  python run_high_roi.py --t-scaling          # Improvement 3
  python run_high_roi.py --returns            # Improvement 4
  python run_high_roi.py --tuning             # Improvement 5
  python run_high_roi.py --all                # Everything
"""

from __future__ import annotations

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from temporal_repr.agents.train import train
from temporal_repr.analysis.collect import collect_hidden_states
from temporal_repr.analysis.cka import linear_cka, generate_engineered_encodings
from temporal_repr.analysis.linear_probe import linear_probe_timestep, linear_probe_phase
from temporal_repr.analysis.pca_vis import pca_analysis
from temporal_repr.analysis.mutual_info import estimate_mi_binning
from temporal_repr.environments.env_factory import make_env
from temporal_repr.agents.lstm_policy import LSTMActorCritic

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "results" / "models"

SEEDS = [42, 123, 456, 789, 1024]


def _load_model(exp_name, seed=42, device="cpu"):
    exp_dir = MODELS_DIR / exp_name
    if not exp_dir.exists():
        seed_dir = MODELS_DIR / f"{exp_name}_seed{seed}"
        if seed_dir.exists():
            exp_dir = seed_dir
        else:
            raise FileNotFoundError(f"No model found for {exp_name} seed={seed}")

    ckpt_path = exp_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = exp_dir / "final.pt"

    env = make_env("doorgrid", mode="temporal", time_encoding="none", seed=seed)
    obs_dim = env.observation_space.shape[0]
    model = LSTMActorCritic(obs_dim, n_actions=4, hidden_dim=64)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model, env


def _collect_data(exp_name, seed=42, n_episodes=200):
    model, env = _load_model(exp_name, seed=seed)
    return collect_hidden_states(model, env, agent_type="lstm", n_episodes=n_episodes)


# ============================================================
# Improvement 1: Bootstrap CI for CKA
# ============================================================

def bootstrap_cka(hidden_states, timesteps, phases, period, max_t,
                  n_bootstrap=1000, max_samples=2000, ci=0.95):
    n = len(timesteps)
    rng = np.random.RandomState(42)

    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        hs = hidden_states[idx]
        ts = timesteps[idx]
        ph = phases[idx]
    else:
        hs, ts, ph = hidden_states, timesteps, phases

    encodings = generate_engineered_encodings(ts, ph, period, max_t)
    enc_names = list(encodings.keys())

    all_cka = {name: [] for name in enc_names}

    for i in range(n_bootstrap):
        boot_idx = rng.choice(len(ts), len(ts), replace=True)
        hs_boot = hs[boot_idx]
        ts_boot = ts[boot_idx]
        ph_boot = ph[boot_idx]

        enc_boot = generate_engineered_encodings(ts_boot, ph_boot, period, max_t)

        for name in enc_names:
            cka_val = linear_cka(hs_boot, enc_boot[name])
            all_cka[name].append(cka_val)

    results = {}
    alpha = (1 - ci) / 2
    for name in enc_names:
        arr = np.array(all_cka[name])
        results[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            f"ci_{ci}_lo": float(np.percentile(arr, alpha * 100)),
            f"ci_{ci}_hi": float(np.percentile(arr, (1 - alpha) * 100)),
            "values": [float(v) for v in arr],
        }

    return results


def run_bootstrap():
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 1: Bootstrap CI for CKA")
    print("=" * 70)

    all_results = {}

    for seed in SEEDS:
        exp_name = f"05_lstm_dg_temp_none_seed{seed}"
        fig_dir = FIGURES_DIR / exp_name

        if not (fig_dir / f"{exp_name}_results.json").exists():
            exp_name = "05_lstm_dg_temp_none"

        print(f"\n  Seed {seed}: collecting data...")
        data = _collect_data("05_lstm_dg_temp_none", seed=seed, n_episodes=200)
        print(f"    {len(data['timesteps'])} samples, period={data['period']}")

        print(f"    Running bootstrap (1000 iterations)...")
        t0 = time.time()
        boot = bootstrap_cka(
            data["hidden_states"], data["timesteps"], data["phases"],
            period=data["period"], max_t=80
        )
        elapsed = time.time() - t0

        for enc, stats in boot.items():
            print(f"    CKA({enc}) = {stats['mean']:.3f} "
                  f"[{stats['ci_0.95_lo']:.3f}, {stats['ci_0.95_hi']:.3f}]")

        all_results[f"seed_{seed}"] = boot
        print(f"    Done in {elapsed:.1f}s")

    save_path = FIGURES_DIR / "bootstrap_cka_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {save_path}")

    print("\n  --- Degeneracy Statistical Test ---")
    sin_pe_per_seed = []
    phase_per_seed = []
    for seed in SEEDS:
        key = f"seed_{seed}"
        if key in all_results:
            sin_pe_per_seed.append(all_results[key].get("sinusoidal_pe", {}).get("mean", 0))
            phase_per_seed.append(all_results[key].get("phase_sin_cos", {}).get("mean", 0))

    sin_pe_arr = np.array(sin_pe_per_seed)
    phase_arr = np.array(phase_per_seed)
    diff = sin_pe_arr - phase_arr
    t_stat = diff.mean() / (diff.std() / np.sqrt(len(diff))) if diff.std() > 0 else float('inf')

    print(f"    CKA(sin_PE) across seeds: {sin_pe_arr.mean():.3f} ± {sin_pe_arr.std():.3f}")
    print(f"    CKA(phase) across seeds:  {phase_arr.mean():.3f} ± {phase_arr.std():.3f}")
    print(f"    Mean difference: {diff.mean():.3f}, t={t_stat:.2f} (N={len(diff)})")

    dominant = []
    for i, seed in enumerate(SEEDS):
        if sin_pe_per_seed[i] > phase_per_seed[i]:
            dominant.append("sin_PE")
        else:
            dominant.append("phase")
    print(f"    Dominant encoding per seed: {dict(zip(SEEDS, dominant))}")

    return all_results


# ============================================================
# Improvement 2: T=8 multi-seed
# ============================================================

def create_doorgrid_t(period):
    from reproduction.experiments.temporal_environments import DoorGridLarge

    class DoorGridT(DoorGridLarge):
        PERIOD = period

        def __init__(self, mode='static', seed=42):
            self.PERIOD = period
            super().__init__(mode=mode, seed=seed)

    DoorGridT.__name__ = f"DoorGridT{period}"
    DoorGridT.__qualname__ = f"DoorGridT{period}"
    return DoorGridT


def run_t8_multiseed():
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 2: T=8 DoorGrid Multi-Seed")
    print("=" * 70)

    from reproduction.experiments.temporal_environments import DoorGridLarge

    DoorGridLarge_orig_PERIOD = DoorGridLarge.PERIOD
    DoorGridLarge.PERIOD = 8

    try:
        all_metrics = {}
        for seed in SEEDS:
            exp_name = f"11_lstm_dg_t8_seed{seed}"
            fig_dir = FIGURES_DIR / exp_name

            if (fig_dir / f"{exp_name}_results.json").exists():
                print(f"\n  Seed {seed}: already exists, loading...")
                with open(fig_dir / f"{exp_name}_results.json") as f:
                    results = json.load(f)
            else:
                print(f"\n  Seed {seed}: training...")
                t0 = time.time()

                model, log = train(
                    env_name="doorgrid", mode="temporal", time_encoding="none",
                    agent_type="lstm", n_episodes=10000, seed=seed,
                    save_dir=str(MODELS_DIR / exp_name), verbose=True
                )
                train_time = time.time() - t0
                print(f"    Training: {train_time:.1f}s")

                eval_env = make_env("doorgrid", mode="temporal",
                                    time_encoding="none", seed=seed + 1000)
                data = collect_hidden_states(model, eval_env, agent_type="lstm",
                                             n_episodes=200)

                from temporal_repr.analysis.report import full_analysis
                results = full_analysis(data, experiment_name=exp_name,
                                        save_dir=str(fig_dir))

                log_summary = {
                    "final_avg_reward": sum(log["episode_rewards"][-100:]) / 100,
                    "best_eval_reward": max(log.get("eval_rewards", [0])),
                    "train_time_s": train_time,
                    "n_episodes": 10000,
                }
                with open(fig_dir / f"{exp_name}_train_log.json", "w") as f:
                    json.dump(log_summary, f, indent=2)

            for key in ['probe_timestep', 'probe_phase', 'pca', 'cka', 'mi_binning']:
                if key in results:
                    if key not in all_metrics:
                        all_metrics[key] = {}
                    for k, v in results[key].items():
                        if isinstance(v, (int, float)):
                            if k not in all_metrics[key]:
                                all_metrics[key][k] = []
                            all_metrics[key][k].append(v)

        stats = {"experiment": "11_lstm_dg_t8_multiseed", "n_seeds": len(SEEDS),
                 "seeds": SEEDS, "period": 8}
        for key in all_metrics:
            stats[key] = {}
            for k, vals in all_metrics[key].items():
                arr = np.array(vals)
                stats[key][k] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "values": [float(v) for v in arr],
                }

        save_path = FIGURES_DIR / "11_lstm_dg_t8_multiseed.json"
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n  T=8 Multi-Seed Results:")
        cka_raw = stats.get("cka", {}).get("raw_scalar", {}).get("values", [])
        cka_sin = stats.get("cka", {}).get("sinusoidal_pe", {}).get("values", [])
        if cka_raw and cka_sin:
            print(f"    CKA(raw_scalar): {np.mean(cka_raw):.3f} ± {np.std(cka_raw):.3f}")
            print(f"    CKA(sin_PE):     {np.mean(cka_sin):.3f} ± {np.std(cka_sin):.3f}")
        print(f"  Saved to {save_path}")

    finally:
        DoorGridLarge.PERIOD = DoorGridLarge_orig_PERIOD

    return stats


# ============================================================
# Improvement 3: T scaling {4, 8, 16, 32}
# ============================================================

def run_t_scaling(periods=None, seeds=None):
    if periods is None:
        periods = [4, 8, 16, 32]
    if seeds is None:
        seeds = SEEDS

    print("\n" + "=" * 70)
    print(f"  IMPROVEMENT 3: T Scaling {{4, 8, 16, 32}}")
    print("=" * 70)

    from reproduction.experiments.temporal_environments import DoorGridLarge

    DoorGridLarge_orig_PERIOD = DoorGridLarge.PERIOD

    all_scaling_results = {}

    for T in periods:
        print(f"\n{'#'*60}")
        print(f"  Period T = {T}")
        print(f"{'#'*60}")

        DoorGridLarge.PERIOD = T

        seed_results = {}
        for seed in seeds:
            exp_name = f"scaling_dg_t{T}_seed{seed}"
            fig_dir = FIGURES_DIR / exp_name

            if (fig_dir / f"{exp_name}_results.json").exists():
                print(f"\n  T={T} seed={seed}: loading existing...")
                with open(fig_dir / f"{exp_name}_results.json") as f:
                    results = json.load(f)
                train_log_path = fig_dir / f"{exp_name}_train_log.json"
                train_log = {}
                if train_log_path.exists():
                    with open(train_log_path) as f:
                        train_log = json.load(f)
            else:
                print(f"\n  T={T} seed={seed}: training...")
                t0 = time.time()

                model, log = train(
                    env_name="doorgrid", mode="temporal", time_encoding="none",
                    agent_type="lstm", n_episodes=10000, seed=seed,
                    save_dir=str(MODELS_DIR / exp_name), verbose=True
                )
                train_time = time.time() - t0

                eval_env = make_env("doorgrid", mode="temporal",
                                    time_encoding="none", seed=seed + 1000)
                data = collect_hidden_states(model, eval_env, agent_type="lstm",
                                             n_episodes=200)

                from temporal_repr.analysis.report import full_analysis
                results = full_analysis(data, experiment_name=exp_name,
                                        save_dir=str(fig_dir))

                train_log = {
                    "final_avg_reward": sum(log["episode_rewards"][-100:]) / 100,
                    "best_eval_reward": max(log.get("eval_rewards", [0])),
                    "train_time_s": train_time,
                    "n_episodes": 10000,
                }
                with open(fig_dir / f"{exp_name}_train_log.json", "w") as f:
                    json.dump(train_log, f, indent=2)

            seed_results[seed] = {
                "results": results,
                "train_log": train_log,
            }

        all_scaling_results[T] = seed_results

        cka_raw_vals = [sr["results"]["cka"]["raw_scalar"] for sr in seed_results.values()]
        cka_sin_vals = [sr["results"]["cka"]["sinusoidal_pe"] for sr in seed_results.values()]
        pc1_corr = [sr["results"]["pca"]["pc1_timestep_corr"] for sr in seed_results.values()]
        r2_vals = [sr["results"]["probe_timestep"]["r2"] for sr in seed_results.values()]

        print(f"\n  T={T} Summary:")
        print(f"    R²(t)       = {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")
        print(f"    CKA(raw)    = {np.mean(cka_raw_vals):.3f} ± {np.std(cka_raw_vals):.3f}")
        print(f"    CKA(sin_PE) = {np.mean(cka_sin_vals):.3f} ± {np.std(cka_sin_vals):.3f}")
        print(f"    PC1-t corr  = {np.mean(pc1_corr):.3f} ± {np.std(pc1_corr):.3f}")

    DoorGridLarge.PERIOD = DoorGridLarge_orig_PERIOD

    scaling_summary = {}
    for T in periods:
        scaling_summary[str(T)] = {}
        for metric_key in ["cka", "pca", "probe_timestep", "probe_phase", "mi_binning"]:
            vals_dict = defaultdict(list)
            for seed in seeds:
                r = all_scaling_results[T][seed]["results"].get(metric_key, {})
                for k, v in r.items():
                    if isinstance(v, (int, float)):
                        vals_dict[k].append(v)
            scaling_summary[str(T)][metric_key] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "values": [float(x) for x in v]}
                for k, v in vals_dict.items()
            }

    save_path = FIGURES_DIR / "t_scaling_summary.json"
    with open(save_path, "w") as f:
        json.dump(scaling_summary, f, indent=2)
    print(f"\n  Saved scaling summary to {save_path}")

    return all_scaling_results


# ============================================================
# Improvement 4: Task Return data
# ============================================================

def collect_returns():
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 4: Task Return Data Collection")
    print("=" * 70)

    return_data = {}

    exp_dirs = sorted(FIGURES_DIR.glob("*"))
    for d in exp_dirs:
        if not d.is_dir():
            continue
        name = d.name
        log_files = list(d.glob("*_train_log.json"))
        result_files = list(d.glob("*_results.json"))

        if log_files:
            with open(log_files[0]) as f:
                log = json.load(f)
            return_data[name] = log

            if result_files:
                with open(result_files[0]) as f:
                    res = json.load(f)
                return_data[name]["r2_timestep"] = res.get("probe_timestep", {}).get("r2", 0)
                return_data[name]["phase_accuracy"] = res.get("probe_phase", {}).get("accuracy", 0)
                return_data[name]["cka_best_encoding"] = max(
                    res.get("cka", {}).items(), key=lambda x: x[1]
                )[0] if res.get("cka") else "N/A"
                return_data[name]["cka_best_score"] = max(
                    res.get("cka", {}).values()
                ) if res.get("cka") else 0

    print(f"\n  {'Experiment':<40} {'Best Eval':>10} {'Final Avg':>10} {'R²(t)':>8} {'CKA':>6}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
    for name, data in sorted(return_data.items()):
        best = data.get("best_eval_reward", "N/A")
        final = data.get("final_avg_reward", "N/A")
        r2 = data.get("r2_timestep", "N/A")
        cka = data.get("cka_best_score", "N/A")
        if isinstance(best, float):
            best = f"{best:.3f}"
        if isinstance(final, float):
            final = f"{final:.3f}"
        if isinstance(r2, float):
            r2 = f"{r2:.3f}"
        if isinstance(cka, float):
            cka = f"{cka:.3f}"
        print(f"  {name:<40} {best:>10} {final:>10} {r2:>8} {cka:>6}")

    save_path = FIGURES_DIR / "task_returns_summary.json"
    with open(save_path, "w") as f:
        json.dump(return_data, f, indent=2)
    print(f"\n  Saved to {save_path}")

    return return_data


# ============================================================
# Improvement 5: Single-neuron tuning curves
# ============================================================

def analyze_tuning_curves():
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 5: Single-Neuron Tuning Curves")
    print("=" * 70)

    all_tuning = {}

    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        data = _collect_data("05_lstm_dg_temp_none", seed=seed, n_episodes=200)
        hs = data["hidden_states"]
        ts = data["timesteps"]
        phases = data["phases"]
        period = data["period"]
        hidden_dim = data["hidden_dim"]

        tuning = np.zeros((hidden_dim, period))
        counts = np.zeros(period)
        for i in range(len(ts)):
            p = phases[i] % period
            tuning[:, p] += hs[i]
            counts[p] += 1
        tuning = tuning / np.maximum(counts, 1)

        tuning_std = np.zeros((hidden_dim, period))
        tuning_sq = np.zeros((hidden_dim, period))
        for i in range(len(ts)):
            p = phases[i] % period
            tuning_sq[:, p] += hs[i] ** 2
        for p in range(period):
            if counts[p] > 1:
                tuning_std[:, p] = np.sqrt(
                    tuning_sq[:, p] / counts[p] - tuning[:, p] ** 2
                )

        phase_selectivity = np.zeros(hidden_dim)
        for d in range(hidden_dim):
            phase_selectivity[d] = (tuning[d].max() - tuning[d].min()) / (
                tuning[d].max() + abs(tuning[d].min()) + 1e-8
            )

        sorted_dims = np.argsort(phase_selectivity)[::-1]

        preferred_phase = np.zeros(hidden_dim, dtype=int)
        for d in range(hidden_dim):
            preferred_phase[d] = np.argmax(np.abs(tuning[d]))

        top_k = 16
        result = {
            "top_time_cell_dims": sorted_dims[:top_k].tolist(),
            "phase_selectivity_top": phase_selectivity[sorted_dims[:top_k]].tolist(),
            "preferred_phase_top": preferred_phase[sorted_dims[:top_k]].tolist(),
            "mean_activity": tuning.mean(axis=1).tolist(),
            "phase_selectivity_all": phase_selectivity.tolist(),
            "tuning_curves_top": {
                str(d): tuning[d].tolist() for d in sorted_dims[:top_k]
            },
        }

        all_tuning[f"seed_{seed}"] = result

        print(f"    Top time cells (dims): {sorted_dims[:8]}")
        print(f"    Phase selectivity:     {phase_selectivity[sorted_dims[:8]].round(3).tolist()}")
        print(f"    Preferred phases:      {preferred_phase[sorted_dims[:8]].tolist()}")

    seeds_have_same_cells = True
    top_sets = []
    for seed_key, tuning_data in all_tuning.items():
        top_sets.append(set(tuning_data["top_time_cell_dims"][:8]))
    if len(top_sets) >= 2:
        overlap = top_sets[0]
        for s in top_sets[1:]:
            overlap = overlap & s
        print(f"\n  Overlap of top-8 time cells across seeds: {len(overlap)}/8")
        print(f"    → {'Same' if len(overlap) == 8 else 'Different'} time cell populations")

    save_path = FIGURES_DIR / "tuning_curves_analysis.json"
    with open(save_path, "w") as f:
        json.dump(all_tuning, f, indent=2)
    print(f"\n  Saved to {save_path}")

    return all_tuning


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--t8-multiseed", action="store_true")
    parser.add_argument("--t-scaling", action="store_true")
    parser.add_argument("--returns", action="store_true")
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.bootstrap, args.t8_multiseed, args.t_scaling,
                args.returns, args.tuning, args.all]):
        args.all = True

    if args.all or args.bootstrap:
        run_bootstrap()

    if args.all or args.returns:
        collect_returns()

    if args.all or args.tuning:
        analyze_tuning_curves()

    if args.all or args.t8_multiseed:
        run_t8_multiseed()

    if args.all or args.t_scaling:
        run_t_scaling()

    print("\n\n" + "=" * 70)
    print("  ALL IMPROVEMENTS COMPLETE")
    print("=" * 70)
