import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

class AmortizedInferenceNet(nn.Module):
    """
    Neural network that maps data to posterior samples.
    """
    def __init__(self, input_size, hidden_size=64, latent_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def sample_posterior(self, x, n_samples=5):
        """
        Sample from the posterior distribution.
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.forward(x)
            samples = []
            for _ in range(n_samples):
                eps = torch.randn_like(mu)
                z = mu + torch.exp(0.5 * logvar) * eps
                samples.append(z)
            samples = torch.stack(samples, dim=1)
        return samples

def prepare_data(returns, macro_df, seq_len=10):
    """
    Prepare sequences for training.
    """
    if isinstance(returns, np.ndarray):
        return None, None
    if len(returns) < seq_len + 1:
        return None, None
    common_idx = returns.index.intersection(macro_df.index)
    if len(common_idx) < seq_len + 1:
        return None, None
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    X, y = [], []
    for i in range(seq_len, len(ret_aligned)):
        ret_seq = ret_aligned.iloc[i-seq_len:i].values.reshape(-1, 1)
        macro_seq = macro_aligned.iloc[i-seq_len:i].values
        seq_features = np.concatenate([ret_seq, macro_seq], axis=1)
        X.append(seq_features)
        y.append(ret_aligned.iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def amortized_inference_score(returns, macro_df, hidden_size=64, latent_dim=16, seq_len=10, epochs=30, batch_size=32, lr=0.001, n_samples=5):
    """
    Train amortized inference network and return posterior mean score.
    """
    X, y = prepare_data(returns, macro_df, seq_len)
    if X is None or len(X) < batch_size:
        return 0.0
    input_size = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Split into train and validation
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    if len(y_val) < 5:
        return 0.0
    model = AmortizedInferenceNet(input_size, hidden_size, latent_dim).to(device)
    # Training: we want the network to predict the posterior distribution of parameters
    # We use a simulated "likelihood-free" setup: we generate synthetic parameters
    # and train the network to recover them.
    # For simplicity, we treat the target y as the parameter of interest.
    # We train the network to predict the distribution of y given X.
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Loss: Gaussian negative log-likelihood (VAE-style)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mu, logvar = model(X_batch)
            # Negative log-likelihood of y under the predicted posterior
            # We use a Gaussian likelihood: y ~ N(mu, exp(logvar))
            # Reconstruction loss
            recon_loss = 0.5 * torch.sum((y_batch - mu)**2 / torch.exp(logvar) + logvar + np.log(2*np.pi))
            # KL divergence (to a standard normal prior)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
            loss = recon_loss + 0.01 * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    # Inference: get posterior samples for the validation set
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        # Get posterior samples
        samples = model.sample_posterior(X_val_t, n_samples)
        # Compute posterior mean
        posterior_mean = samples.mean(dim=1).cpu().numpy()
        # Score = average posterior mean over validation set
        score = np.mean(posterior_mean)
    return float(score)
