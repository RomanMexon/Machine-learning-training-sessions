import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    r = np.random.uniform(size=data.shape[1])

    for step in range(num_steps):
        r = data.dot(r) / ((np.sum(data.dot(r) ** 2)) ** 0.5)
    l = (r.T.dot(data)).dot(r) / (r.T.dot(r))

    return float(l), r