"""Run experiments with multiple seeds and collect statistics."""
import sys; sys.path.insert(0, '.')
import json, numpy as np
from pathlib import Path
from temporal_repr.experiments.run_training import run_experiment

SEEDS = [42, 123, 456, 789, 1024]
KEY_EXPERIMENTS = [
    ('01_lstm_fl_temp_none',  'frozenlake',    'temporal', 'none',       'lstm', 10000),
    ('05_lstm_dg_temp_none',  'doorgrid',      'temporal', 'none',       'lstm', 10000),
    ('06_lstm_sk_temp_none',  'sokoban_gate',  'temporal', 'none',       'lstm', 15000),
    ('09_mlp_fl_temp_none',   'frozenlake',    'temporal', 'none',       'mlp',  10000),
]

for exp_args in KEY_EXPERIMENTS:
    name = exp_args[0]
    all_metrics = {}

    for seed_idx, seed in enumerate(SEEDS):
        suffix = f"_seed{seed}"
        seeded_name = f"{name}{suffix}"
        modified_args = (seeded_name,) + exp_args[1:5] + (exp_args[5],) + (seed,)

        print(f"\n{'='*60}")
        print(f"  {name} seed={seed} ({seed_idx+1}/{len(SEEDS)})")
        print(f"{'='*60}")

        results = run_experiment(*modified_args)

        # Extract key metrics
        for key in ['probe_timestep', 'probe_phase', 'pca', 'cka', 'mi_binning']:
            if key in results:
                if key not in all_metrics:
                    all_metrics[key] = {}
                for k, v in results[key].items():
                    if isinstance(v, (int, float)):
                        if k not in all_metrics[key]:
                            all_metrics[key][k] = []
                        all_metrics[key][k].append(v)

    # Compute statistics
    stats = {"experiment": name, "n_seeds": len(SEEDS), "seeds": SEEDS}
    for key in all_metrics:
        stats[key] = {}
        for k, vals in all_metrics[key].items():
            arr = np.array(vals)
            stats[key][k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": [float(v) for v in arr],
            }

    # Save
    save_path = Path(f"temporal_repr/figures/{name}_multiseed.json")
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved multi-seed stats to {save_path}")
    print(f"  R²(t) = {stats['probe_timestep']['r2']['mean']:.3f} ± {stats['probe_timestep']['r2']['std']:.3f}")
    pc1 = stats.get('pca', {}).get('pc1_explained_variance', {})
    if pc1:
        print(f"  PC1% = {pc1['mean']:.1%} ± {pc1['std']:.1%}")
