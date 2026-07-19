# Likelihood-Free Amortized Inference for ETFs

Trains a neural network to directly output posterior samples given any dataset, without running MCMC. Once trained, inference is instantaneous – ideal for online settings where models must be repeatedly fit to rolling windows. The per‑ETF score is the posterior mean of the expected return.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- LSTM-based amortized inference network
- VAE-style training for posterior distribution
- Instantaneous inference (no MCMC)
- Score = posterior mean (higher = stronger signal)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-likelihood-free-amortized-inference-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to neural net training)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High posterior mean → expected upward move.
- Low posterior mean → expected downward move.

## Requirements

See `requirements.txt`.
