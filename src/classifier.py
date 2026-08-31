import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import os

class FraudClassifier(nn.Module):
    """
    MLP binary classifier for fraud detection.
    """
    def __init__(self, input_dim=29):
        super(FraudClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, X):
        """
        Predict probabilities.

        Args:
            X: numpy array or torch tensor

        Returns:
            numpy array of probabilities
        """
        self.eval()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            probs = torch.sigmoid(self(X)).squeeze(-1).numpy()
        return probs

    def predict(self, X, threshold=0.5):
        """
        Predict binary labels.

        Args:
            X: numpy array or torch tensor
            threshold: classification threshold

        Returns:
            numpy array of binary labels
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def find_optimal_threshold(self, X_val, y_val):
        """
        Find the classification threshold that maximises F1 on the validation set.
        Searches [0.05, 0.95] in steps of 0.05.

        Using 0.5 on a ~0.1% fraud stream almost always predicts all-negative.
        The optimal threshold is typically much lower (0.1-0.3).
        """
        from sklearn.metrics import f1_score
        probs = self.predict_proba(X_val)
        best_f1, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.05, 0.95, 0.05):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(thresh)
        return best_thresh

def train_classifier(classifier, X_train, y_train, X_val, y_val, epochs=30, batch_size=256, lr=1e-3, patience=5, device='cpu'):
    """
    Train the classifier with early stopping.

    Args:
        classifier: FraudClassifier model
        X_train, y_train: training data
        X_val, y_val: validation data
        epochs: max epochs
        batch_size: batch size
        lr: learning rate
        patience: early stopping patience
        device: torch device

    Returns:
        trained classifier
    """
    classifier.to(device)

    # Calculate pos_weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_f1 = 0
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Training Classifier"):
        classifier.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = classifier(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        classifier.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = classifier(X_batch).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        val_f1 = f1_score(val_targets, (np.array(val_preds) >= 0.5).astype(int))

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            tqdm.write(f"Early stopping at epoch {epoch+1}")
            break

        tqdm.write(f"Epoch {epoch+1}/{epochs}, Val F1: {val_f1:.4f}")

    return classifier

if __name__ == "__main__":
    # Test classifier
    classifier = FraudClassifier()
    print(classifier)
    # Dummy data
    X_train = np.random.randn(1000, 29)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 29)
    y_val = np.random.randint(0, 2, 200)
    trained_clf = train_classifier(classifier, X_train, y_train, X_val, y_val, epochs=5, device='cpu')
    probs = trained_clf.predict_proba(X_val[:10])
    preds = trained_clf.predict(X_val[:10])
    print(f"Probabilities: {probs}")
    print(f"Predictions: {preds}")