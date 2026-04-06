import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Compute KL divergence between q(z|x) and N(0, I).
    """

    DKL = -(1/2) * np.sum(1 + log_var - np.power(mu, 2) - np.exp(log_var))

    return DKL