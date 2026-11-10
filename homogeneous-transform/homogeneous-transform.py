import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """

    T_arr = np.array(T)
    if isinstance(points[0], list):
        L = len(points)
    else:
        L = 1
        points = [points]
        
    points_arr = np.hstack((np.array(points), np.ones((L, 1))))
    print(points_arr)
    dotp = (T_arr @ points_arr.T).T
    xyz = dotp[:, :3]

    if len(xyz) == 1:
        return xyz[0]
    else:
        return xyz