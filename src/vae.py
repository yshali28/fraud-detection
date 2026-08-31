import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os

class VAE(nn.Module):
    """
    Variational Autoencoder for tabular data.
    """
    def __init__(self, input_dim=29, hidden_dim1=64, hidden_dim2=32, latent_dim=16):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim2, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

def vae_loss(recon, x, mu, log_var, beta=0.5):
    """
    VAE loss: reconstruction loss + KL divergence
    """
    mse_loss = nn.MSELoss()(recon, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss /= x.size(0)  # Average over batch
    total_loss = mse_loss + beta * kl_loss
    return total_loss, mse_loss, kl_loss

def train_vae(vae, data, epochs=50, batch_size=256, lr=1e-3, device='cpu', save_path=None):
    """
    Train VAE on given data.

    Args:
        vae: VAE model
        data: numpy array of shape (n_samples, input_dim)
        epochs: number of epochs
        batch_size: batch size
        lr: learning rate
        device: torch device
        save_path: path to save best model

    Returns:
        trained vae
    """
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Split into train/val 80/20
    n_samples = data.shape[0]
    n_train = int(0.8 * n_samples)
    train_data = data[:n_train]
    val_data = data[n_train:]

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')

    for epoch in tqdm(range(epochs), desc="Training VAE"):
        vae.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, log_var = vae(x)
            loss, _, _ = vae_loss(recon, x, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon, mu, log_var = vae(x)
                loss, _, _ = vae_loss(recon, x, mu, log_var)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(vae.state_dict(), save_path)

        tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return vae

def generate_samples(vae, n_samples, device='cpu'):
    """
    Generate synthetic samples from VAE.

    Args:
        vae: trained VAE
        n_samples: number of samples to generate
        device: torch device

    Returns:
        numpy array of generated samples
    """
    vae.to(device)
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, vae.latent_dim).to(device)
        samples = vae.decode(z).cpu().numpy()
    return samples

if __name__ == "__main__":
    # Test VAE
    vae = VAE()
    print(vae)
    # Dummy data
    dummy_data = np.random.randn(1000, 29)
    trained_vae = train_vae(vae, dummy_data, epochs=5, device='cpu')
    samples = generate_samples(trained_vae, 100)
    print(f"Generated samples shape: {samples.shape}")