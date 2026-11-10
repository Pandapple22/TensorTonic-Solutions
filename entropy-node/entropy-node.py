import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    if not y:
        return 0.0
    label_count = {}
    N = len(y)
    H = 0
    for y_i in y:
        label_count[y_i] = label_count.get(y_i, 0) + 1

    for k, v in label_count.items():
        if v == 0:
            continue
        p = v / N
        H -= p*np.log2(p)
    
    return H