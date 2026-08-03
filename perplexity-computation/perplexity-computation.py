def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    import numpy as np
    
    H = 0
    for dist, token in zip(prob_distributions, actual_tokens):
        H -= np.log(dist[token])

    return np.exp(H/len(actual_tokens))