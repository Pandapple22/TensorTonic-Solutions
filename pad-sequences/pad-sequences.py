import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return None
        
    if not max_len:
        max_len = max([len(seq) for seq in seqs])

    padded_seqs = []
    for seq in seqs:
        pad_length = max_len - len(seq)
        if pad_length > 0:
            seq += [pad_value for _ in range(pad_length)]
        elif pad_length < 0:
            seq = seq[:pad_length]
        padded_seqs.append(seq)

    ans = np.array(padded_seqs)
    return ans