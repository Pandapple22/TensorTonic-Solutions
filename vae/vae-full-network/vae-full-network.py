import numpy as np

class VAE:
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize VAE.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Initialize weights here
        self.W, self.b = np.random.rand(latent_dim, input_dim), np.random.rand(latent_dim, 1)
        
        self.W_mu, self.b_mu = np.random.rand(latent_dim, latent_dim), np.random.rand(latent_dim, 1)
        
        self.W_log_var, self.b_log_var = np.random.rand(latent_dim, latent_dim), np.random.rand(latent_dim, 1)
        
        self.W_dec, self.b_dec = np.random.rand(input_dim, latent_dim), np.random.rand(input_dim, 1)
        
    
    def forward(self, x: np.ndarray) -> tuple:
        """
        Full forward pass through VAE.
        """
        h = self.W @ x.T + self.b
        mu = self.W_mu @ h + self.b_mu
        log_var = self.W_log_var @ h + self.b_log_var
        
        mu, log_var = mu.T, log_var.T
        std = np.exp(0.5 * log_var)

        eps = np.random.normal(size=mu.shape)
        z = mu + std * eps
        
        x_recon = self.W_dec @ z.T + self.b_dec
        x_recon = x_recon.T

        return x_recon, mu, log_var
        
        
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples from prior.
        """

        z = np.random.normal(size=(n_samples, self.latent_dim))
        x_recon = self.W_dec @ z.T + self.b_dec

        return x_recon.T
        

        
        