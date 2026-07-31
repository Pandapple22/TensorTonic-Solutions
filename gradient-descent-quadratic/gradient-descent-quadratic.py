def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x= x0
    for s in range(steps):
        y = 2*a*x + b
        x -= lr*y
    return x