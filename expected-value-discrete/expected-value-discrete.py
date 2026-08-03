import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    
    x_arr = np.array(x)
    p_arr = np.array(p)

    if sum(p_arr) != 1:
        raise ValueError

    return sum(x_arr * p_arr)
