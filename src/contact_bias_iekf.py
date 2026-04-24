import numpy as np
from dataclasses import dataclass

from .dynamics import IEKFDynamics
from .types import IMUMeasurement


# Error-state index slices for the 21-dim error vector:
#   [delta_phi, delta_v, delta_p, delta_d_L, delta_d_R, delta_b_L, delta_b_R]
IDX_PHI = slice(0, 3)
IDX_V   = slice(3, 6)
IDX_P   = slice(6, 9)
IDX_DL  = slice(9, 12)
IDX_DR  = slice(12, 15)
IDX_BL  = slice(15, 18)
IDX_BR  = slice(18, 21)
DIM_ERR = 21


@dataclass
class ContactBiasState:
    """Augmented state for the contact-bias IEKF.

    X:   (7, 7)  SE_4(3) matrix [R | v | p | d_L | d_R]
    b_L: (3,)    left-foot position bias (world frame)
    b_R: (3,)    right-foot position bias (world frame)
    P:   (21,21) covariance of the error state
    """
    X: np.ndarray
    b_L: np.ndarray
    b_R: np.ndarray
    P: np.ndarray


class ContactBiasIEKF:
    """Right-invariant EKF with per-foot contact-position biases.

    Augments the SE_4(3) state with world-frame biases b_L, b_R in R^3,
    modeled as random walks. Bias is added to the contact position before
    rotation to body frame:

        h_body = R^T (d_i + b_i - p)

    Uses the right-invariant world-frame innovation
    nu_world = R_hat (y_body - h_body), whose first-order linearization
    gives H = [0, 0, -I, +I_{d_i}, +I_{b_i}] up to an O(|b| |phi_err|)
    cross term -skew(delta_phi) b_hat that is dropped.

    Composes IEKFDynamics for the SE_4(3) part; biases are carried
    separately so the base module is unchanged.
    """

    def __init__(self, gravity=None, noise_params=None):
        self.base = IEKFDynamics(gravity=gravity, noise_params=noise_params)
        self.noise = self.base.noise

    # ----- state construction -----

    def make_state(self, R=None, v=None, p=None, dl=None, dr=None,
                   b_L=None, b_R=None, P=None):
        """Build a ContactBiasState."""
        X = self.base.make_state(R=R, v=v, p=p, dl=dl, dr=dr)
        if b_L is None:
            b_L = np.zeros(3)
        if b_R is None:
            b_R = np.zeros(3)
        if P is None:
            P = np.eye(DIM_ERR) * 1e-6
        return ContactBiasState(
            X=X,
            b_L=np.asarray(b_L, dtype=float).ravel(),
            b_R=np.asarray(b_R, dtype=float).ravel(),
            P=np.asarray(P, dtype=float),
        )

    # ----- propagation -----

    def propagate(self, state, gyro_bias, accel_bias, imu: IMUMeasurement, dt):
        """Propagate (X, b, P) forward by dt using an IMU measurement.

        X is propagated by the base IEKFDynamics (IMU integration on SE_4(3)).
        Biases follow a random walk: mean stays put, covariance grows via Q.
        Covariance update is P_new = F P F^T + Q.

        Args:
            state: current ContactBiasState.
            gyro_bias: (3,) gyroscope bias.
            accel_bias: (3,) accelerometer bias.
            imu: IMUMeasurement with gyro and accel fields.
            dt: time step in seconds.

        Returns:
            Propagated ContactBiasState.
        """
        X_new = self.base.propagate_state(
            state.X, gyro_bias, accel_bias, imu, dt
        )
        F = self.compute_F(dt)
        Q = self.compute_Q(dt)
        P_new = F @ state.P @ F.T + Q
        return ContactBiasState(
            X=X_new,
            b_L=state.b_L.copy(),
            b_R=state.b_R.copy(),
            P=P_new,
        )

    # ----- linearization -----

    def compute_F(self, dt):
        """21x21 discrete state transition.

        Upper-left 15x15 is the base SE_4(3) F. Lower-right 6x6 is identity
        (pure random walk on biases). Off-diagonals are zero: contact bias
        does not enter propagation of any other state.
        """
        F = np.eye(DIM_ERR)
        F[:15, :15] = self.base.compute_F(dt)
        return F

    def compute_Q(self, dt):
        """21x21 discrete process noise covariance.

        Q[:15, :15] is the base Q. Q[b_L, b_L] = Q[b_R, b_R] = bias_cov * dt.
        """
        Q = np.zeros((DIM_ERR, DIM_ERR))
        Q[:15, :15] = self.base.compute_Q(dt)
        Q[IDX_BL, IDX_BL] = self.noise.bias_cov * dt
        Q[IDX_BR, IDX_BR] = self.noise.bias_cov * dt
        return Q

    # ----- measurement model -----

    def build_H(self, foot="left"):
        """Measurement Jacobian (3, 21) for the world-frame innovation.

        Args:
            foot: 'left' or 'right'.

        Returns:
            (3, 21) Jacobian matrix.
        """
        H = np.zeros((3, DIM_ERR))
        H[:, IDX_P] = -np.eye(3)
        if foot == "left":
            H[:, IDX_DL] = np.eye(3)
            H[:, IDX_BL] = np.eye(3)
        elif foot == "right":
            H[:, IDX_DR] = np.eye(3)
            H[:, IDX_BR] = np.eye(3)
        else:
            raise ValueError(f"foot must be 'left' or 'right', got {foot!r}")
        return H

    def predict_measurement(self, state, foot="left"):
        """Expected body-frame FK measurement h_body(X, b) = R^T (d + b - p)."""
        R, _, p, dl, dr = self.base.unpack_state(state.X)
        if foot == "left":
            return R.T @ (dl + state.b_L - p)
        if foot == "right":
            return R.T @ (dr + state.b_R - p)
        raise ValueError(f"foot must be 'left' or 'right', got {foot!r}")

    def innovation(self, state, y_fk_body, foot="left"):
        """World-frame innovation from a body-frame FK measurement.

        Returns nu_world = R_hat @ (y_body - h_body(X_hat, b_hat)). This
        matches the state-independent H from build_H().
        """
        y = np.asarray(y_fk_body, dtype=float).ravel()
        R_hat = state.X[:3, :3]
        nu_body = y - self.predict_measurement(state, foot)
        return R_hat @ nu_body

    # ----- contact switching -----

    def reset_contact(self, state, foot, d_world, d_cov=None, b_cov=None):
        """Re-initialize d_i and b_i on a new foot contact.

        Sets d_i in X from the supplied world position, resets b_i to zero,
        and zeros the row/col of P for both blocks before installing a
        loose diagonal prior. Default prior is (1 cm)^2 per axis for both
        d and b, reflecting that the new contact point and any prior bias
        are uncorrelated with the current state.

        Args:
            state: current ContactBiasState.
            foot: 'left' or 'right'.
            d_world: (3,) initial world-frame contact position (from FK).
            d_cov: (3,3) initial covariance of d_i (default (1 cm)^2 I).
            b_cov: (3,3) initial covariance of b_i (default (1 cm)^2 I).

        Returns:
            ContactBiasState with the block reset.
        """
        if d_cov is None:
            d_cov = np.eye(3) * 1e-4
        if b_cov is None:
            b_cov = np.eye(3) * 1e-4

        X = state.X.copy()
        P = state.P.copy()
        b_L = state.b_L.copy()
        b_R = state.b_R.copy()

        if foot == "left":
            X[:3, 5] = np.asarray(d_world, dtype=float).ravel()
            _zero_row_col(P, IDX_DL)
            _zero_row_col(P, IDX_BL)
            P[IDX_DL, IDX_DL] = d_cov
            P[IDX_BL, IDX_BL] = b_cov
            b_L = np.zeros(3)
        elif foot == "right":
            X[:3, 6] = np.asarray(d_world, dtype=float).ravel()
            _zero_row_col(P, IDX_DR)
            _zero_row_col(P, IDX_BR)
            P[IDX_DR, IDX_DR] = d_cov
            P[IDX_BR, IDX_BR] = b_cov
            b_R = np.zeros(3)
        else:
            raise ValueError(f"foot must be 'left' or 'right', got {foot!r}")

        return ContactBiasState(X=X, b_L=b_L, b_R=b_R, P=P)


def _zero_row_col(P, idx):
    """Zero out the given rows and columns of P in place."""
    P[idx, :] = 0.0
    P[:, idx] = 0.0
