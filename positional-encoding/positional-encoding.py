import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for d in range(d_model):
            if d % 2 == 0:
                PE[pos, d] = np.sin(pos / base**(d/d_model))
            else:
                PE[pos, d] = np.cos(pos / base**((d-1)/d_model))
            
    return PE
            