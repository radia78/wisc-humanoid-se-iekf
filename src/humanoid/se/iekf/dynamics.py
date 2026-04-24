import numpy as np
from humanoid.se.iekf.types import IMUMeasurement, NoiseParams
from humanoid.se.iekf.lie_group import skew, so3_exp


class IEKFDynamics:
    """Right-invariant EKF state propagation (Section 5 of Hartley et al.).

    State X is a (3+k) x (3+k) SE_k(3) matrix.
    With 4 extra columns (velocity, position, left_contact, right_contact) this is
    SE_4(3), giving a 7x7 state matrix.

    Covariance dimension: 3 + 3*4 = 15.
    """

    def __init__(self, gravity=None, noise_params=None):
        self.g = np.array([0, 0, -9.81]) if gravity is None else np.asarray(gravity)
        self.noise = noise_params if noise_params is not None else NoiseParams()

    @staticmethod
    def quart_to_rot(q: np.array):
        """
        Convert quarternion coordinates to a rotation matrix for default state estimation
        """
        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]

        R = np.diag(np.array([
            qw**2 + qx**2 - qy**2 - qz**2,
            qw**2 - qx**2 + qy**2 - qz**2,
            qw**2 - qx**2 - qy**2 + qz**2
        ]))

        R[0, 1] = 2 * (qx * qy - qw * qz)
        R[0, 2] = 2 * (qw * qy + qx * qz)
        R[1, 0] = 2 * (qx * qy + qw * qz)
        R[1, 2] = 2 * (qy * qz - qw * qx)
        R[2, 0] = 2 * (qx * qz - qw * qy)
        R[2, 1] = 2 * (qw * qx + qy * qz)

        return R


    @staticmethod
    def make_state(R=None, v=None, p=None, dl=None, dr=None):
        """Build a 7x7 SE_4(3) state matrix.

        Columns: [R | v | p | d_left | d_right]
        """
        X = np.eye(7)
        if R is not None:
            X[:3, :3] = np.asarray(R)
        if v is not None:
            X[:3, 3] = np.asarray(v).ravel()
        if p is not None:
            X[:3, 4] = np.asarray(p).ravel()
        if dl is not None:
            X[:3, 5] = np.asarray(dl).ravel()
        if dr is not None:
            X[:3, 6] = np.asarray(dr).ravel()
        return X

    @staticmethod
    def unpack_state(X):
        """Extract R, v, p, d_left, d_right from 7x7 state."""
        R = X[:3, :3]
        v = X[:3, 3]
        p = X[:3, 4]
        dl = X[:3, 5]
        dr = X[:3, 6]
        return R, v, p, dl, dr

    # ----- propagation -----

    def propagate_state(self, X, gyro_bias, accel_bias, imu: IMUMeasurement, dt):
        """Propagate state X forward by dt using IMU measurement.

        Uses first-order Euler discretization matching the paper:
          R_new = R @ Exp((gyro - bg) * dt)
          v_new = v + (R @ (accel - ba) + g) * dt
          p_new = p + v * dt + 0.5 * (R @ (accel - ba) + g) * dt^2
          contacts unchanged

        Args:
            X: (7,7) current state.
            gyro_bias: (3,) gyroscope bias.
            accel_bias: (3,) accelerometer bias.
            imu: IMUMeasurement with gyro and accel fields.
            dt: time step in seconds.

        Returns:
            X_new: (7,7) propagated state.
        """
        R, v, p, dl, dr = self.unpack_state(X)

        omega = imu.gyro - gyro_bias  # corrected angular velocity
        alpha = imu.accel - accel_bias  # corrected acceleration

        R_new = R @ so3_exp(omega * dt)
        a_world = R @ alpha + self.g  # acceleration in world frame
        v_new = v + a_world * dt
        p_new = p + v * dt + 0.5 * a_world * dt**2

        return self.make_state(R_new, v_new, p_new, dl, dr)

    # ----- linearization -----

    def compute_F(self, dt):
        """Compute discrete state transition matrix F (15x15).

        For right-invariant EKF, F depends only on gravity (state-independent).
        F = I + A*dt where A is the continuous-time system matrix.

        The error state is [delta_phi, delta_v, delta_p, delta_d1, delta_d2].
        """
        F = np.eye(15)
        # dv/dphi: gravity cross-product effect
        F[3:6, 0:3] = skew(self.g) * dt
        # dp/dv
        F[6:9, 3:6] = np.eye(3) * dt
        # dp/dphi (second order from gravity)
        F[6:9, 0:3] = 0.5 * skew(self.g) * dt**2
        return F

    def compute_Q(self, dt):
        """Compute discrete process noise covariance Q (15x15).

        Q_d = F_noise @ Q_c @ F_noise^T * dt
        Simplified: diagonal blocks for gyro, accel, and contact noise.
        """
        Q = np.zeros((15, 15))
        # Rotation noise (gyro)
        Q[0:3, 0:3] = self.noise.gyro_cov * dt
        # Velocity noise (accel)
        Q[3:6, 3:6] = self.noise.accel_cov * dt
        # Position noise (small, from accel integration)
        Q[6:9, 6:9] = self.noise.accel_cov * dt**3 / 3.0
        # Contact point noise
        Q[9:12, 9:12] = self.noise.contact_cov * dt
        Q[12:15, 12:15] = self.noise.contact_cov * dt
        return Q
