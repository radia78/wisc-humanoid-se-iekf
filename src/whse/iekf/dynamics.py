import numpy as np
from whse.iekf.types import IMUMeasurement, NoiseParams, RobotState
from whse.iekf.lie_group import (
    skew,
    sek3_adjoint,
    so3_exp,
    so3_left_jacobian,
    so3_gamma_2,
)

"""
The base implementation follows the implementation specified in section 5 of Hartley, et al, 2019
"""


class RIEKFDynamics:
    def __init__(self, dt=None, gravity=None, noise_params=None):
        self.g = np.array([0, 0, -9.81]) if gravity is None else np.asarray(gravity)
        self.noise = noise_params if noise_params is not None else NoiseParams()
        self.dt = dt if dt is not None else 0.02  # Default Mujoco time delay

        # Constant matrices that is time invariant
        self.A = self._build_A()
        self.Q_k = self._build_cov()
        # As specified, this is an approximate of matrix exponentials (Taylor First Order Approximation)
        self.Phi = self._build_phi()

    def propagate(self, robot_state: RobotState, imu_data: IMUMeasurement):
        # Easier naming conventions
        X = robot_state.X
        P = robot_state.P

        # Obtain the data from IMU sensors
        omega = imu_data.gyro
        a = imu_data.accel

        # Compute the state propogation
        R, v, p, dl, dr = robot_state.unpack_state()
        R_new = R @ so3_exp(omega * self.dt)
        v_new = v + (R @ so3_left_jacobian(omega * self.dt) @ a + self.g) * self.dt
        p_new = (
            p
            + v * self.dt
            + (R @ so3_gamma_2(omega * self.dt) @ a + 0.5 * self.g) * self.dt**2
        )

        # Compute the covariance propagation
        Phi_adj = self.Phi @ sek3_adjoint(X, k=4)
        Q_k_hat = Phi_adj @ self.Q_k @ Phi_adj.T * self.dt
        P_new = self.Phi @ P @ self.Phi.T + Q_k_hat

        return robot_state.make_state(R_new, v_new, p_new, dl, dr), P_new

    def _build_cov(self):
        # Compute the covariance matrix
        cov = np.zeros((15, 15))
        # Rotation noise (gyro)
        cov[0:3, 0:3] = self.noise.gyro_cov
        # Velocity noise (accel)
        cov[3:6, 3:6] = self.noise.accel_cov
        # Contact point noise
        cov[9:12, 9:12] = self.noise.contact_cov
        # Contact point noise
        cov[12:15, 12:15] = self.noise.contact_cov

        return cov

    def _build_A(self):
        # Compute the A matrix
        A = np.zeros((15, 15))
        A[3:6, 0:3] = skew(self.g)
        A[6:9, 3:6] = np.eye(3)

        return A
    
    def _build_phi(self):
        phi = np.eye(15, 15)
        phi[3:6, 0:3] = skew(self.g) * self.dt
        phi[6:9, 0:3] = 0.5 * skew(self.g) * self.dt ** 2
        phi[6:9, 3:6] = np.eye(3) * self.dt

        return phi

