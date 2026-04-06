import numpy as np

def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.
    """
    
    W, b =  np.random.rand(output_dim, z.shape[1]), np.random.rand(output_dim, 1)
    h = W @ z.T + b

    return h.T