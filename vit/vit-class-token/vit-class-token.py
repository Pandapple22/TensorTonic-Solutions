import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    """
    cls_token = np.random.rand(1, embed_dim)
    cls_token = cls_token[None, :, :]
    cls_token = np.repeat(cls_token, patches.shape[0], axis=0)
    seq = np.concatenate([cls_token, patches], axis=1)

    return seq