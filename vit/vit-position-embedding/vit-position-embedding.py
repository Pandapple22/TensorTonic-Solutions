import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int) -> np.ndarray:
    """
    Add learnable position embeddings to patch embeddings.
    """
    pos_emb = np.random.rand(*patches.shape)
    patches = patches + pos_emb
    return patches