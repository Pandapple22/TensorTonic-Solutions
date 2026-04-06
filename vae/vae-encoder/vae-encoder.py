import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    """
    Encode input to latent distribution parameters.
    """

    W, b = np.random.rand(latent_dim, x.shape[1]), np.random.rand(latent_dim, 1)
    h = W @ x.T + b

    W_mu, b_mu = np.random.rand(latent_dim, latent_dim), np.random.rand(latent_dim, 1)
    W_logvar, b_logvar = np.random.rand(latent_dim, latent_dim), np.random.rand(latent_dim, 1)
    mu = W_mu @ h + b_mu
    logvar = W_logvar @ h + b_logvar

    return mu.T, logvar.T