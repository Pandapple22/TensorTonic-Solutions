import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    if not A:
        return []
    
    N, M = len(A), len(A[0])
    A_T = np.zeros((M, N))
    for i in range(N):
        for j in range(M):
            A_T[j, i] = A[i][j]
    return A_T
