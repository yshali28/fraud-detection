import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_data(data_path):
    """
    Load the credit card fraud detection dataset, perform temporal splitting,
    normalize Amount, and return the eras.

    Args:
        data_path (str): Path to creditcard.csv

    Returns:
        list: List of 4 DataFrames, one for each era
    """
    # Load data
    df = pd.read_csv(data_path)

    # Sort by Time column
    df = df.sort_values('Time').reset_index(drop=True)

    # Split into 4 temporal eras of equal size
    n_samples = len(df)
    era_size = n_samples // 4

    eras = []
    for i in range(4):
        start_idx = i * era_size
        if i == 3:  # Last era takes the remainder
            end_idx = n_samples
        else:
            end_idx = (i + 1) * era_size
        era_df = df.iloc[start_idx:end_idx].copy()
        eras.append(era_df)

    # Normalize Amount using StandardScaler fit on Era 0 only
    scaler = StandardScaler()
    scaler.fit(eras[0][['Amount']])

    for i in range(4):
        eras[i]['Amount'] = scaler.transform(eras[i][['Amount']])
        # Drop Time column
        eras[i] = eras[i].drop(columns=['Time'])

    # Print class distributions
    for i, era in enumerate(eras):
        class_counts = era['Class'].value_counts()
        print(f"Era {i} class distribution:")
        print(f"  Legit (0): {class_counts.get(0, 0)}")
        print(f"  Fraud (1): {class_counts.get(1, 0)}")
        print(f"  Fraud ratio: {class_counts.get(1, 0) / len(era):.4f}")
        print()

    return eras

if __name__ == "__main__":
    # For testing
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'creditcard.csv')
    eras = load_and_preprocess_data(data_path)
    print(f"Loaded {len(eras)} eras")
    print(f"Features: {list(eras[0].columns)}")