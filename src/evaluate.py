import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import os

def compute_backward_transfer(results_df):
    """
    Compute backward transfer for each system.

    Args:
        results_df: DataFrame with results

    Returns:
        dict: BT per system
    """
    bt_dict = {}
    for system in results_df['system'].unique():
        system_df = results_df[results_df['system'] == system]
        era0_after_era0 = system_df[(system_df['era'] == 0) & (system_df['eval_on_era'] == 0)]['f1'].values[0]
        era0_after_era3 = system_df[(system_df['era'] == 0) & (system_df['eval_on_era'] == 3)]['f1'].values[0]
        bt = era0_after_era3 - era0_after_era0
        bt_dict[system] = bt
    return bt_dict

def plot_f1_per_era(results_df, save_path):
    """
    Plot F1 score per era for all systems.
    """
    plt.figure(figsize=(10, 6))
    systems = results_df['system'].unique()
    for system in systems:
        system_df = results_df[results_df['system'] == system]
        # Get F1 for each era when evaluated after training on that era
        f1s = []
        for era in range(4):
            f1 = system_df[(system_df['era'] == era) & (system_df['eval_on_era'] == era)]['f1'].values[0]
            f1s.append(f1)
        plt.plot(range(4), f1s, marker='o', label=system)
    plt.xlabel('Era')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Era')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_auc_per_era(results_df, save_path):
    """
    Plot AUC per era for all systems.
    """
    plt.figure(figsize=(10, 6))
    systems = results_df['system'].unique()
    for system in systems:
        system_df = results_df[results_df['system'] == system]
        aucs = []
        for era in range(4):
            auc = system_df[(system_df['era'] == era) & (system_df['eval_on_era'] == era)]['auc'].values[0]
            aucs.append(auc)
        plt.plot(range(4), aucs, marker='o', label=system)
    plt.xlabel('Era')
    plt.ylabel('AUC-ROC')
    plt.title('AUC-ROC per Era')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_backward_transfer(bt_dict, save_path):
    """
    Bar plot of backward transfer.
    """
    systems = list(bt_dict.keys())
    bts = list(bt_dict.values())
    plt.figure(figsize=(8, 6))
    bars = plt.bar(systems, bts, color=['blue', 'orange', 'green', 'red'])
    plt.ylabel('Backward Transfer')
    plt.title('Backward Transfer (F1 on Era 0 after Era 3 - F1 on Era 0 after Era 0)')
    plt.axhline(0, color='black', linestyle='--')
    for bar, bt in zip(bars, bts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bt:.4f}', ha='center', va='bottom')
    plt.savefig(save_path)
    plt.close()

def plot_retraining_events(ours_events, baseline_c_events, save_path):
    """
    Bar plot of retraining events.
    """
    plt.figure(figsize=(6, 6))
    systems = ['Ours', 'Baseline_C']
    events = [ours_events, baseline_c_events]
    bars = plt.bar(systems, events, color=['red', 'green'])
    plt.ylabel('Number of Retraining Events')
    plt.title('Retraining Events')
    for bar, event in zip(bars, events):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(event), ha='center', va='bottom')
    plt.savefig(save_path)
    plt.close()

def plot_tsne_vae_samples(real_era0, vae_samples, save_path):
    """
    t-SNE plot of real Era 0 vs VAE-generated samples.
    """
    combined = np.vstack([real_era0, vae_samples])
    labels = ['Real'] * len(real_era0) + ['VAE'] * len(vae_samples)
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(combined)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=labels, alpha=0.7)
    plt.title('t-SNE: Real Era 0 vs VAE-Generated Samples')
    plt.savefig(save_path)
    plt.close()

def save_results(results_df, save_path):
    """
    Save results to CSV.
    """
    results_df.to_csv(save_path, index=False)

def print_summary_table(results_df, bt_dict, ours_events, baseline_c_events):
    """
    Print final summary table.
    """
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'System':<15} {'F1 Era0':<10} {'F1 Era1':<10} {'F1 Era2':<10} {'F1 Era3':<10} {'BT':<10} {'Retrains':<10}")
    print("-"*80)
    systems = ['Ours', 'Baseline_A', 'Baseline_B', 'Baseline_C']
    for system in systems:
        system_df = results_df[results_df['system'] == system]
        f1s = []
        for era in range(4):
            f1 = system_df[(system_df['era'] == era) & (system_df['eval_on_era'] == era)]['f1'].values[0]
            f1s.append(f"{f1:.4f}")
        bt = bt_dict.get(system, 0)
        retrains = ours_events if system == 'Ours' else (baseline_c_events if system == 'Baseline_C' else 0)
        print(f"{system:<15} {' | '.join(f1s)} {bt:<10.4f} {retrains:<10}")
    print("="*80)

if __name__ == "__main__":
    # Dummy test
    results = [
        {'system': 'Ours', 'era': 0, 'eval_on_era': 0, 'f1': 0.8, 'precision': 0.8, 'recall': 0.8, 'auc': 0.85, 'backward_transfer': 0.8},
        {'system': 'Ours', 'era': 0, 'eval_on_era': 3, 'f1': 0.75, 'precision': 0.75, 'recall': 0.75, 'auc': 0.8, 'backward_transfer': 0.75},
    ]
    results_df = pd.DataFrame(results)
    bt_dict = compute_backward_transfer(results_df)
    print(bt_dict)