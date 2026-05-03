import numpy as np


def skew(v):
    """3-vector -> 3x3 skew-symmetric matrix."""
    v = np.asarray(v, dtype=float).ravel()
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def unskew(S):
    """3x3 skew-symmetric matrix -> 3-vector."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def so3_exp(phi):
    """Exponential map: axis-angle (3,) -> rotation matrix (3,3).

    Uses Rodrigues' formula.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3)
    S = skew(phi)
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * S
        + ((1 - np.cos(theta)) / theta**2) * S @ S
    )


def so3_log(R):
    """Logarithmic map: rotation matrix (3,3) -> axis-angle (3,)."""
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return unskew(R - R.T) / 2
    return (theta / (2 * np.sin(theta))) * unskew(R - R.T)


def so3_left_jacobian(phi):
    """Left Jacobian of SO(3) for axis-angle phi (3,)."""
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3)
    S = skew(phi)
    return (
        np.eye(3)
        + ((1 - np.cos(theta)) / theta**2) * S
        + ((theta - np.sin(theta)) / theta**3) * S @ S
    )


def so3_gamma_2(phi):
    """Gamma 2 specified in the IEKF paper"""
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3)
    S = skew(phi)
    return (
        0.5 * np.eye(3)
        + ((theta - np.sin(theta)) / theta**3) * S
        + ((theta**2 + 2 * np.cos(theta) - 2) / (2 * theta**4)) * S @ S
    )


def _build_sek3_matrix(R, columns):
    """Build (3+k) x (3+k) SE_k(3) matrix from R and k column vectors."""
    k = len(columns)
    n = 3 + k
    X = np.eye(n)
    X[:3, :3] = R
    for i, col in enumerate(columns):
        X[:3, 3 + i] = np.asarray(col, dtype=float).ravel()
    return X


def sek3_exp(xi, k):
    """Exponential map for SE_k(3).

    xi is a vector of length 3 + 3k:
      [phi (3), v1 (3), v2 (3), ..., vk (3)]
    Returns (3+k) x (3+k) matrix.
    """
    xi = np.asarray(xi, dtype=float).ravel()
    phi = xi[:3]
    R = so3_exp(phi)
    J = so3_left_jacobian(phi)
    columns = []
    for i in range(k):
        vi = xi[3 + 3 * i : 3 + 3 * (i + 1)]
        columns.append(J @ vi)
    return _build_sek3_matrix(R, columns)


def sek3_log(X, k):
    """Logarithmic map for SE_k(3).

    X is (3+k) x (3+k) matrix.
    Returns vector of length 3 + 3k.
    """
    R = X[:3, :3]
    phi = so3_log(R)
    J = so3_left_jacobian(phi)
    J_inv = np.linalg.solve(J, np.eye(3))
    xi = np.zeros(3 + 3 * k)
    xi[:3] = phi
    for i in range(k):
        col = X[:3, 3 + i]
        xi[3 + 3 * i : 3 + 3 * (i + 1)] = J_inv @ col
    return xi


def sek3_adjoint(X, k):
    """Adjoint representation of SE_k(3).

    X is (3+k) x (3+k).
    Returns (3+3k) x (3+3k) adjoint matrix.
    """
    n = 3 + 3 * k
    R = X[:3, :3]
    Ad = np.zeros((n, n))
    Ad[:3, :3] = R
    for i in range(k):
        col = X[:3, 3 + i]
        Ad[3 + 3 * i : 3 + 3 * (i + 1), :3] = skew(col) @ R
        Ad[3 + 3 * i : 3 + 3 * (i + 1), 3 + 3 * i : 3 + 3 * (i + 1)] = R
    return Ad


def sek3_inverse(X, k):
    """Inverse of SE_k(3) matrix.

    X is (3+k) x (3+k).
    Returns (3+k) x (3+k) inverse.
    """
    n = 3 + k
    R = X[:3, :3]
    X_inv = np.eye(n)
    X_inv[:3, :3] = R.T
    for i in range(k):
        col = X[:3, 3 + i]
        X_inv[:3, 3 + i] = -R.T @ col
    return X_inv
