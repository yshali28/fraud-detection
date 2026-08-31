import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from classifier import FraudClassifier, train_classifier
from vae import VAE, train_vae, generate_samples
from drift_detector import FraudDriftDetector
import os
from tqdm import tqdm
import torch

def continual_learning_system(eras, device='cpu', seed=42):
    """
    Our continual learning system with drift-triggered VAE replay.

    Args:
        eras: list of 4 era DataFrames
        device: torch device
        seed: random seed

    Returns:
        dict: results per era, and retraining events
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []
    replay_buffer = {}  # vae_fraud and vae_nonfraud per era
    retraining_events = 0

    for era_idx in range(4):
        print(f"Continual Learning - Era {era_idx}")
        current_data = eras[era_idx]
        X_current = current_data.drop('Class', axis=1).values
        y_current = current_data['Class'].values

        if era_idx == 0:
            # Cold start
            fraud_data = X_current[y_current == 1]
            nonfraud_data = X_current[y_current == 0]

            vae_fraud = VAE()
            vae_nonfraud = VAE()

            if len(fraud_data) > 0:
                vae_fraud = train_vae(vae_fraud, fraud_data, device=device)
            if len(nonfraud_data) > 0:
                vae_nonfraud = train_vae(vae_nonfraud, nonfraud_data, device=device)

            replay_buffer[0] = {'fraud': vae_fraud, 'nonfraud': vae_nonfraud}

            # Train classifier on Era 0
            n_train = int(0.8 * len(X_current))
            X_train, X_val = X_current[:n_train], X_current[n_train:]
            y_train, y_val = y_current[:n_train], y_current[n_train:]

            classifier = FraudClassifier()
            classifier = train_classifier(classifier, X_train, y_train, X_val, y_val, device=device)
            threshold = classifier.find_optimal_threshold(X_val, y_val)
            print(f"  Era 0 optimal threshold: {threshold:.2f}")

            log_msg = "Era 0: Cold start"

        else:
            # Check for drift using previous classifier on current era
            detector = FraudDriftDetector()
            for i in range(len(X_current)):
                prob = classifier.predict_proba(X_current[i:i+1])[0]
                pred = 1 if prob >= threshold else 0
                error = 1 if pred != y_current[i] else 0
                detector.update(error, fraud_prob=float(prob))

            drift_detected = detector.drift_detected

            if drift_detected:
                # Retrain with replay
                retraining_events += 1

                # Build replay set
                # Use fixed counts: 500 fraud + 2000 non-fraud per past era.
                # Proportional sampling would give ~0 fraud samples (ratio ~0.1%)
                # which defeats the purpose of replay for rare fraud patterns.
                N_REPLAY_FRAUD = 500
                N_REPLAY_NONFRAUD = 2000
                synthetic_data = []
                for past_era in range(era_idx):
                    vaes = replay_buffer[past_era]

                    if len(eras[past_era][eras[past_era]['Class'] == 1]) > 0:
                        synth_fraud = generate_samples(vaes['fraud'], N_REPLAY_FRAUD, device)
                        synthetic_data.append((synth_fraud, np.ones(N_REPLAY_FRAUD)))
                    synth_nonfraud = generate_samples(vaes['nonfraud'], N_REPLAY_NONFRAUD, device)
                    synthetic_data.append((synth_nonfraud, np.zeros(N_REPLAY_NONFRAUD)))

                # Combine and shuffle before splitting to avoid all-synthetic train / all-real val
                X_synth_list = [s[0] for s in synthetic_data]
                y_synth_list = [s[1] for s in synthetic_data]
                if X_synth_list:
                    X_synth = np.vstack(X_synth_list)
                    y_synth = np.concatenate(y_synth_list)
                    X_combined = np.vstack([X_synth, X_current])
                    y_combined = np.concatenate([y_synth, y_current])
                else:
                    X_combined = X_current
                    y_combined = y_current

                shuffle_idx = np.random.permutation(len(X_combined))
                X_combined, y_combined = X_combined[shuffle_idx], y_combined[shuffle_idx]

                n_train_split = int(0.8 * len(X_combined))
                X_train_split = X_combined[:n_train_split]
                X_val = X_combined[n_train_split:]
                y_train_split = y_combined[:n_train_split]
                y_val = y_combined[n_train_split:]

                classifier = FraudClassifier()
                classifier = train_classifier(classifier, X_train_split, y_train_split, X_val, y_val, device=device)
                threshold = classifier.find_optimal_threshold(X_val, y_val)
                print(f"  Era {era_idx} optimal threshold: {threshold:.2f}")

                log_msg = f"Era {era_idx}: Drift detected, retrained"

            else:
                # No drift detected — do a light fine-tune on current era only
                # so the model doesn't go stale as fraud patterns shift subtly.
                n_train = int(0.8 * len(X_current))
                X_ft_train, X_ft_val = X_current[:n_train], X_current[n_train:]
                y_ft_train, y_ft_val = y_current[:n_train], y_current[n_train:]
                classifier = train_classifier(
                    classifier, X_ft_train, y_ft_train, X_ft_val, y_ft_val,
                    epochs=10, device=device
                )
                threshold = classifier.find_optimal_threshold(X_ft_val, y_ft_val)
                print(f"  Era {era_idx} optimal threshold (fine-tune): {threshold:.2f}")
                log_msg = f"Era {era_idx}: No drift, light fine-tune only"

            # Train new VAEs on current era
            fraud_data = X_current[y_current == 1]
            nonfraud_data = X_current[y_current == 0]

            vae_fraud = VAE()
            vae_nonfraud = VAE()

            if len(fraud_data) > 0:
                vae_fraud = train_vae(vae_fraud, fraud_data, device=device)
            if len(nonfraud_data) > 0:
                vae_nonfraud = train_vae(vae_nonfraud, nonfraud_data, device=device)

            replay_buffer[era_idx] = {'fraud': vae_fraud, 'nonfraud': vae_nonfraud}

        print(log_msg)

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
                'system': 'Ours',
                'era': eval_era,
                'eval_on_era': era_idx,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'backward_transfer': f1 if eval_era == 0 else None
            })

    return results, retraining_events

if __name__ == "__main__":
    # Dummy test
    eras = []
    for i in range(4):
        df = pd.DataFrame({
            'V' + str(j): np.random.randn(100) for j in range(28)
        })
        df['Amount'] = np.random.randn(100)
        df['Class'] = np.random.randint(0, 2, 100)
        eras.append(df)
    results, events = continual_learning_system(eras, device='cpu')
    print(f"Retraining events: {events}")
    print(results[:5])