import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from classifier import FraudClassifier, train_classifier
from vae import VAE, train_vae, generate_samples
import os
from tqdm import tqdm

def baseline_a_no_replay(eras, device='cpu', seed=42):
    """
    Baseline A: No Replay - retrain classifier only on current era data.

    Args:
        eras: list of 4 era DataFrames
        device: torch device
        seed: random seed

    Returns:
        dict: results per era
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []

    for era_idx in range(4):
        print(f"Baseline A - Era {era_idx}")
        current_data = eras[era_idx]
        X = current_data.drop('Class', axis=1).values
        y = current_data['Class'].values

        # Split train/val 80/20
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        # Train classifier
        classifier = FraudClassifier()
        classifier = train_classifier(classifier, X_train, y_train, X_val, y_val, device=device)
        threshold = classifier.find_optimal_threshold(X_val, y_val)

        # Evaluate on all eras seen so far
        for eval_era in range(era_idx + 1):
            eval_data = eras[eval_era]
            X_eval = eval_data.drop('Class', axis=1).values
            y_eval = eval_data['Class'].values

            probs = classifier.predict_proba(X_eval)
            preds = (probs >= threshold).astype(int)

            f1 = f1_score(y_eval, preds)
            precision = precision_score(y_eval, preds, zero_division=0)
            recall = recall_score(y_eval, preds, zero_division=0)
            auc = roc_auc_score(y_eval, probs)

            results.append({
                'system': 'Baseline_A',
                'era': eval_era,
                'eval_on_era': era_idx,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'backward_transfer': f1 if eval_era == 0 else None  # Will compute later
            })

    return results

def baseline_b_raw_replay(eras, device='cpu', seed=42):
    """
    Baseline B: Raw Data Replay - retrain on all real data seen so far.

    Args:
        eras: list of 4 era DataFrames
        device: torch device
        seed: random seed

    Returns:
        dict: results per era
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []
    all_data_seen = []

    for era_idx in range(4):
        print(f"Baseline B - Era {era_idx}")
        current_data = eras[era_idx]
        all_data_seen.append(current_data)

        # Concat all seen data
        combined_data = pd.concat(all_data_seen, ignore_index=True)
        X = combined_data.drop('Class', axis=1).values
        y = combined_data['Class'].values

        # Split train/val 80/20
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        # Train classifier
        classifier = FraudClassifier()
        classifier = train_classifier(classifier, X_train, y_train, X_val, y_val, device=device)
        threshold = classifier.find_optimal_threshold(X_val, y_val)

        # Evaluate on all eras seen so far
        for eval_era in range(era_idx + 1):
            eval_data = eras[eval_era]
            X_eval = eval_data.drop('Class', axis=1).values
            y_eval = eval_data['Class'].values

            probs = classifier.predict_proba(X_eval)
            preds = (probs >= threshold).astype(int)

            f1 = f1_score(y_eval, preds)
            precision = precision_score(y_eval, preds, zero_division=0)
            recall = recall_score(y_eval, preds, zero_division=0)
            auc = roc_auc_score(y_eval, probs)

            results.append({
                'system': 'Baseline_B',
                'era': eval_era,
                'eval_on_era': era_idx,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'backward_transfer': f1 if eval_era == 0 else None
            })

    return results

def baseline_c_vae_always(eras, device='cpu', seed=42):
    """
    Baseline C: VAE Replay Always-On - always replay with VAE samples.

    Args:
        eras: list of 4 era DataFrames
        device: torch device
        seed: random seed

    Returns:
        dict: results per era
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []
    replay_buffer = {}  # vae_fraud and vae_nonfraud per era

    for era_idx in range(4):
        print(f"Baseline C - Era {era_idx}")
        current_data = eras[era_idx]
        X = current_data.drop('Class', axis=1).values
        y = current_data['Class'].values

        # Train VAEs on current era
        fraud_data = X[y == 1]
        nonfraud_data = X[y == 0]

        vae_fraud = VAE()
        vae_nonfraud = VAE()

        if len(fraud_data) > 0:
            vae_fraud = train_vae(vae_fraud, fraud_data, device=device)
        if len(nonfraud_data) > 0:
            vae_nonfraud = train_vae(vae_nonfraud, nonfraud_data, device=device)

        replay_buffer[era_idx] = {'fraud': vae_fraud, 'nonfraud': vae_nonfraud}

        # Build replay set from all previous VAEs
        synthetic_data = []
        for past_era in range(era_idx):
            vaes = replay_buffer[past_era]
            # Generate proportional
            n_fraud = int(500 * (len(eras[past_era][eras[past_era]['Class'] == 1]) / len(eras[past_era])))
            n_nonfraud = 2000 - n_fraud

            if len(eras[past_era][eras[past_era]['Class'] == 1]) > 0:
                synth_fraud = generate_samples(vaes['fraud'], n_fraud, device)
                synthetic_data.append(synth_fraud)
            synth_nonfraud = generate_samples(vaes['nonfraud'], n_nonfraud, device)
            synthetic_data.append(synth_nonfraud)

        # Combine synthetic + real current
        if synthetic_data:
            X_synth = np.vstack(synthetic_data)
            y_synth = np.concatenate([
                np.ones(sum(len(s) for s in synthetic_data if len(s) <= n_fraud)) if n_fraud > 0 else np.array([]),
                np.zeros(sum(len(s) for s in synthetic_data) - (n_fraud if n_fraud > 0 else 0))
            ])
            X_train = np.vstack([X_synth, X])
            y_train = np.concatenate([y_synth, y])
        else:
            X_train = X
            y_train = y

        # Split train/val 80/20
        n_samples = len(X_train)
        n_train = int(0.8 * n_samples)
        X_train_split, X_val = X_train[:n_train], X_train[n_train:]
        y_train_split, y_val = y_train[:n_train], y_train[n_train:]

        # Train classifier
        classifier = FraudClassifier()
        classifier = train_classifier(classifier, X_train_split, y_train_split, X_val, y_val, device=device)
        threshold = classifier.find_optimal_threshold(X_val, y_val)

        # Evaluate on all eras seen so far
        for eval_era in range(era_idx + 1):
            eval_data = eras[eval_era]
            X_eval = eval_data.drop('Class', axis=1).values
            y_eval = eval_data['Class'].values

            probs = classifier.predict_proba(X_eval)
            preds = (probs >= threshold).astype(int)

            f1 = f1_score(y_eval, preds)
            precision = precision_score(y_eval, preds, zero_division=0)
            recall = recall_score(y_eval, preds, zero_division=0)
            auc = roc_auc_score(y_eval, probs)

            results.append({
                'system': 'Baseline_C',
                'era': eval_era,
                'eval_on_era': era_idx,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'backward_transfer': f1 if eval_era == 0 else None
            })

    return results

if __name__ == "__main__":
    # Dummy test
    import torch
    eras = []
    for i in range(4):
        df = pd.DataFrame({
            'V' + str(j): np.random.randn(100) for j in range(28)
        })
        df['Amount'] = np.random.randn(100)
        df['Class'] = np.random.randint(0, 2, 100)
        eras.append(df)
    results = baseline_a_no_replay(eras, device='cpu')
    print(results[:5])