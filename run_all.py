"""
Reproduce all experiments and analyses for the paper.

Usage:
    conda activate rlenv
    python run_all.py --phase train          # Train all agents
    python run_all.py --phase analyze        # Run analysis pipeline
    python run_all.py --phase figures        # Generate figures
    python run_all.py --phase all            # Run everything
"""

import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}")
    subprocess.run(cmd, shell=True, check=True)


def train():
    run("python experiments/run_training.py")
    run("python experiments/run_multiseed.py")
    run("python experiments/run_untrained_control.py")
    run("python experiments/run_high_roi.py")


def analyze():
    run("python experiments/compare_results.py")


def figures():
    run("python paper/generate_figures.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "analyze", "figures", "all"], required=True)
    args = parser.parse_args()

    if args.phase in ("train", "all"):
        train()
    if args.phase in ("analyze", "all"):
        analyze()
    if args.phase in ("figures", "all"):
        figures()
