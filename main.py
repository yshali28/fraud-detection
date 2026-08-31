import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_and_preprocess_data
from baselines import baseline_a_no_replay, baseline_b_raw_replay, baseline_c_vae_always
from continual_learner import continual_learning_system
from evaluate import (
    compute_backward_transfer, plot_f1_per_era, plot_auc_per_era,
    plot_backward_transfer, plot_retraining_events, plot_tsne_vae_samples,
    save_results, print_summary_table
)
from vae import generate_samples
import random

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def auto_detect_device():
    """Auto-detect CUDA or CPU."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    set_seed(args.seed)
    device = args.device if args.device != 'auto' else auto_detect_device()
    print(f"Using device: {device}")

    print("Loading and preprocessing data...")
    eras = load_and_preprocess_data(args.data_path)

    all_results = []

    if not args.skip_baselines:
        print("Running Baseline A: No Replay...")
        results_a = baseline_a_no_replay(eras, device=device, seed=args.seed)
        all_results.extend(results_a)

        print("Running Baseline B: Raw Data Replay...")
        results_b = baseline_b_raw_replay(eras, device=device, seed=args.seed)
        all_results.extend(results_b)

        print("Running Baseline C: VAE Always-On...")
        results_c = baseline_c_vae_always(eras, device=device, seed=args.seed)
        all_results.extend(results_c)

    print("Running Our Continual Learning System...")
    results_ours, retraining_events = continual_learning_system(eras, device=device, seed=args.seed)
    all_results.extend(results_ours)

    results_df = pd.DataFrame(all_results)

    bt_dict = compute_backward_transfer(results_df)

    print("Generating plots...")
    plot_f1_per_era(results_df, os.path.join(args.output_dir, 'plots', 'f1_per_era.png'))
    plot_auc_per_era(results_df, os.path.join(args.output_dir, 'plots', 'auc_per_era.png'))
    plot_backward_transfer(bt_dict, os.path.join(args.output_dir, 'plots', 'backward_transfer.png'))

    if not args.skip_baselines:
        baseline_c_events = 3
    else:
        baseline_c_events = 0
    plot_retraining_events(retraining_events, baseline_c_events, os.path.join(args.output_dir, 'plots', 'retraining_events.png'))

    real_era0 = eras[0].drop('Class', axis=1).values[:1000]
    vae_samples = np.random.randn(1000, 29)
    plot_tsne_vae_samples(real_era0, vae_samples, os.path.join(args.output_dir, 'plots', 'tsne_vae_samples.png'))

    save_results(results_df, os.path.join(args.output_dir, 'results', 'results.csv'))

    print_summary_table(results_df, bt_dict, retraining_events, baseline_c_events)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Fraud Detection via Continual Learning")
    parser.add_argument('--data_path', type=str, default='data/raw/creditcard.csv',
                        help='Path to creditcard.csv')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip_baselines', action='store_true',
                        help='Skip running baselines')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')

    args = parser.parse_args()
    main(args)