import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    """
    Convert image to patch embeddings.
    """
    B, H, W, C = image.shape
    num_patch =  (H*W) // (patch_size*patch_size)
    patches = image.reshape(B, num_patch, -1)
    W_embed, b_embed = np.random.rand(B, embed_dim, patches.shape[-1]), np.random.rand(B, embed_dim, 1)
    embed = W_embed @ np.transpose(patches, (0, 2, 1)) + b_embed

    return np.transpose(embed, (0, 2, 1))