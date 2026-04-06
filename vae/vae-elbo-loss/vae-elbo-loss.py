import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss.
    """

    recon_loss = np.linalg.norm(x - x_recon) ** 2
    kl_loss = -(1/2) * np.sum(1 + log_var - mu**2 - np.exp(log_var))
    loss = recon_loss + kl_loss

    return {"total": loss, "recon": recon_loss, "kl": kl_loss}