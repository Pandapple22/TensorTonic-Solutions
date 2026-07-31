import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    s = np.array(s)
    w = np.array(w)
    
    new_s = beta*s + (1-beta)*np.power(g, 2)
    new_w = w - (lr / (np.sqrt(new_s + eps))) * g

    return new_w, new_s