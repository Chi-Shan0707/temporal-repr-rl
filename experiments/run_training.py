"""
experiments/run_training.py — Train all agent × env × encoding combinations.

Experiment matrix:
  1. LSTM + FrozenLake temporal + no time       → Can LSTM learn phase?
  2. LSTM + FrozenLake temporal + phase sin/cos → Does explicit time help?
  3. LSTM + FrozenLake temporal + raw scalar    → Is raw scalar enough?
  4. LSTM + FrozenLake static + no time         → Control: no temporal structure
  5. LSTM + DoorGrid temporal + no time         → Cyclic-only structure
  6. LSTM + SokobanGate temporal + no time      → Irreversible + periodic
  7. LSTM + SokobanGate static + no time        → Control
  8. MLP + FrozenLake temporal + phase sin/cos  → MLP with explicit time
  9. MLP + FrozenLake temporal + no time        → MLP without time
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "/mnt/d/CS/ReinforcementLearning/undo_gap/temporal_repr")
sys.path.insert(0, "/mnt/d/CS/ReinforcementLearning/undo_gap")

from temporal_repr.agents.train import train
from temporal_repr.analysis.collect import collect_hidden_states
from temporal_repr.analysis.report import full_analysis


BASE_DIR = Path("/mnt/d/CS/ReinforcementLearning/undo_gap/temporal_repr")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

EXPERIMENTS = [
    # (name, env_name, mode, time_encoding, agent_type, n_episodes)
    ("01_lstm_fl_temp_none",      "frozenlake",    "temporal", "none",       "lstm", 10000),
    ("02_lstm_fl_temp_phase",     "frozenlake",    "temporal", "phase",      "lstm", 10000),
    ("03_lstm_fl_temp_raw",       "frozenlake",    "temporal", "raw",        "lstm", 10000),
    ("04_lstm_fl_static_none",    "frozenlake",    "static",   "none",       "lstm", 10000),
    ("05_lstm_dg_temp_none",      "doorgrid",      "temporal", "none",       "lstm", 10000),
    ("06_lstm_sk_temp_none",      "sokoban_gate",  "temporal", "none",       "lstm", 15000),
    ("07_lstm_sk_static_none",    "sokoban_gate",  "static",   "none",       "lstm", 10000),
    ("08_mlp_fl_temp_phase",      "frozenlake",    "temporal", "phase",      "mlp",  10000),
    ("09_mlp_fl_temp_none",       "frozenlake",    "temporal", "none",       "mlp",  10000),
]


def run_experiment(name, env_name, mode, time_encoding, agent_type, n_episodes, seed=42):
    """Run one experiment: train → collect → analyze."""
    print(f"\n{'#'*70}")
    print(f"  Experiment: {name}")
    print(f"  Env={env_name}, Mode={mode}, Time={time_encoding}, Agent={agent_type}, Seed={seed}")
    print(f"{'#'*70}")

    model_dir = RESULTS_DIR / "models" / name
    hs_dir = RESULTS_DIR / "hidden_states" / name
    fig_dir = FIGURES_DIR / name

    # Train
    t0 = time.time()
    model, log = train(
        env_name=env_name,
        mode=mode,
        time_encoding=time_encoding,
        agent_type=agent_type,
        n_episodes=n_episodes,
        seed=seed,
        save_dir=str(model_dir),
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"  Training took {train_time:.1f}s")

    # Create env for collection (different seed from training for generalization)
    from temporal_repr.environments.env_factory import make_env
    eval_seed = seed + 1000  # distinct from training seed
    env = make_env(env_name, mode=mode, time_encoding=time_encoding, seed=eval_seed)

    # Collect hidden states
    print(f"  Collecting hidden states...")
    data = collect_hidden_states(model, env, agent_type=agent_type, n_episodes=200)

    # Run full analysis
    results = full_analysis(data, experiment_name=name, save_dir=str(fig_dir))

    # Save training log
    import json
    log_summary = {
        "final_avg_reward": sum(log["episode_rewards"][-100:]) / 100,
        "best_eval_reward": max(log.get("eval_rewards", [0])),
        "train_time_s": train_time,
        "n_episodes": n_episodes,
    }
    with open(fig_dir / f"{name}_train_log.json", "w") as f:
        json.dump(log_summary, f, indent=2)

    return results


def run_all(experiments=None, skip_existing=True):
    """Run all experiments."""
    all_results = {}
    exps = experiments or EXPERIMENTS

    for exp in exps:
        name = exp[0]
        fig_dir = FIGURES_DIR / name

        if skip_existing and (fig_dir / f"{name}_results.json").exists():
            print(f"\n  Skipping {name} (already done)")
            import json
            with open(fig_dir / f"{name}_results.json") as f:
                all_results[name] = json.load(f)
            continue

        results = run_experiment(*exp)
        all_results[name] = results

    # Print summary table
    print_summary(all_results)

    return all_results


def print_summary(results: dict):
    """Print comparison table of all experiments."""
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"  {'Experiment':<30} {'R²(t)':>6} {'Acc(phase)':>10} {'PC1-r':>6} {'Centroid':>8} {'Best CKA':>20}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*6} {'-'*8} {'-'*20}")

    for name, res in results.items():
        r2 = res.get("probe_timestep", {}).get("r2", 0)
        acc = res.get("probe_phase", {}).get("accuracy", 0)
        pc1r = res.get("pca", {}).get("pc1_timestep_corr", 0)
        centroid = res.get("fourier", {}).get("spectral_centroid", 0)
        cka_scores = res.get("cka", {})
        best_cka = max(cka_scores, key=cka_scores.get) if cka_scores else "N/A"
        best_score = cka_scores.get(best_cka, 0) if cka_scores else 0

        print(f"  {name:<30} {r2:>6.3f} {acc:>10.3f} {pc1r:>6.3f} {centroid:>8.4f} {best_cka:>15.3f} {best_score:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, nargs="+", help="Experiment indices (0-based)")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing")
    args = parser.parse_args()

    if args.all:
        run_all(skip_existing=not args.no_skip)
    elif args.exp is not None:
        exps = [EXPERIMENTS[i] for i in args.exp]
        run_all(experiments=exps, skip_existing=False)
    else:
        # Default: run first experiment as test
        print("Running experiment 0 as test. Use --all for full run.")
        run_all(experiments=[EXPERIMENTS[0]], skip_existing=False)
