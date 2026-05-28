"""
Run full CyclicCorridor experiment: 5 seeds + untrained control + analysis.
"""
import sys, json, time
sys.path.insert(0, "/mnt/d/CS/ReinforcementLearning/undo_gap/temporal_repr")
sys.path.insert(0, "/mnt/d/CS/ReinforcementLearning/undo_gap")

from pathlib import Path
import numpy as np
import torch
from temporal_repr.agents.train import train, evaluate
from temporal_repr.agents.lstm_policy import LSTMActorCritic
from temporal_repr.environments.env_factory import make_env
from temporal_repr.analysis.collect import collect_hidden_states
from temporal_repr.analysis.report import full_analysis

BASE_DIR = Path("/mnt/d/CS/ReinforcementLearning/undo_gap/temporal_repr")
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "results" / "models"
SEEDS = [42, 123, 456, 789, 1024]

# ============================================================
# 1. Train 5 seeds
# ============================================================
print("=" * 60)
print("  PHASE 1: Train 5 seeds on CyclicCorridor")
print("=" * 60)

for seed in SEEDS:
    exp_name = f"20_lstm_cc_temp_none_seed{seed}"
    model_dir = MODELS_DIR / exp_name
    fig_dir = FIGURES_DIR / exp_name

    if (model_dir / "best.pt").exists():
        print(f"\nseed{seed}: model exists, skip training")
        continue

    print(f"\nseed{seed}: training...")
    t0 = time.time()
    model, log = train(
        env_name="cyclic_corridor", mode="temporal", time_encoding="none",
        agent_type="lstm", n_episodes=5000, seed=seed,
        save_dir=str(model_dir), verbose=True
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s")

    # Save train log
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_log = {
        "final_avg_reward": float(np.mean(log["episode_rewards"][-100:])),
        "best_eval_reward": float(max(log.get("eval_rewards", [-999]))),
        "train_time_s": elapsed,
        "n_episodes": 5000,
    }
    with open(fig_dir / f"{exp_name}_train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

# ============================================================
# 2. Untrained control
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 2: Untrained LSTM control")
print("=" * 60)

untrained_dir = FIGURES_DIR / "untrained_lstm_cyclic_corridor"
untrained_results_path = untrained_dir / "untrained_lstm_cyclic_corridor_results.json"

if not untrained_results_path.exists():
    env = make_env("cyclic_corridor", mode="temporal", time_encoding="none", seed=42)
    obs_dim = env.observation_space.shape[0]
    untrained_model = LSTMActorCritic(obs_dim, n_actions=2, hidden_dim=64)
    untrained_model.eval()

    eval_env = make_env("cyclic_corridor", mode="temporal", time_encoding="none", seed=1042)
    data = collect_hidden_states(untrained_model, eval_env, agent_type="lstm", n_episodes=200)
    results = full_analysis(data, experiment_name="untrained_lstm_cyclic_corridor",
                            save_dir=str(untrained_dir))
    print(f"  Untrained R2: {results.get('probe_timestep', {}).get('r2', 'N/A')}")
else:
    print("  Untrained results exist, skip.")

# ============================================================
# 3. Full analysis on trained seeds
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 3: Full analysis on trained seeds")
print("=" * 60)

for seed in SEEDS:
    exp_name = f"20_lstm_cc_temp_none_seed{seed}"
    fig_dir = FIGURES_DIR / exp_name
    model_dir = MODELS_DIR / exp_name

    if (fig_dir / f"{exp_name}_results.json").exists():
        print(f"\nseed{seed}: results exist, skip analysis")
        continue

    print(f"\nseed{seed}: collecting + analyzing...")
    env = make_env("cyclic_corridor", mode="temporal", time_encoding="none", seed=seed)
    obs_dim = env.observation_space.shape[0]
    model = LSTMActorCritic(obs_dim, n_actions=2, hidden_dim=64)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True))
    model.eval()

    eval_env = make_env("cyclic_corridor", mode="temporal", time_encoding="none", seed=seed + 1000)
    data = collect_hidden_states(model, eval_env, agent_type="lstm", n_episodes=200)
    results = full_analysis(data, experiment_name=exp_name, save_dir=str(fig_dir))
    print(f"  R2: {results.get('probe_timestep', {}).get('r2', 'N/A')}")
    print(f"  CKA best: {max(results.get('cka', {}).values()) if results.get('cka') else 'N/A'}")

# ============================================================
# 4. Summary
# ============================================================
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

cka_sin, cka_raw, returns = [], [], []
for seed in SEEDS:
    exp_name = f"20_lstm_cc_temp_none_seed{seed}"
    fig_dir = FIGURES_DIR / exp_name
    results_path = fig_dir / f"{exp_name}_results.json"
    log_path = fig_dir / f"{exp_name}_train_log.json"
    if results_path.exists():
        r = json.load(open(results_path))
        cka = r.get("cka", {})
        cka_sin.append(cka.get("sinusoidal_pe", 0))
        cka_raw.append(cka.get("raw_scalar", 0))
    if log_path.exists():
        tl = json.load(open(log_path))
        returns.append(tl.get("best_eval_reward", 0))

print(f"  CKA(sin_PE): {np.mean(cka_sin):.3f} +/- {np.std(cka_sin):.3f}")
print(f"  CKA(raw):    {np.mean(cka_raw):.3f} +/- {np.std(cka_raw):.3f}")
print(f"  Returns:     {returns}")

print("\n  DONE!")
