import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """

    x_array = np.array(x)
    if rng:
        dropout_array =  (rng.random(x_array.shape) >= p).astype(int) * (1 / (1 - p))
    else:
        dropout_array = (np.random.random(x_array.shape) <= p).astype(int) * (1 / (1 - p))

    return x_array * dropout_array, dropout_array