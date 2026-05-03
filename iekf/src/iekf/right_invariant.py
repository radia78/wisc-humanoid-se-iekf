import numpy as np
from iekf.kinematics import ForwardKinematics
from iekf.utils.lie_group import (
    skew,
    sek3_adjoint,
    so3_exp,
    so3_left_jacobian,
    sek3_exp,
    so3_gamma_2,
)
from iekf.utils.types import RobotState, IMUMeasurement, NoiseParams


class RIEKF:
    def __init__(
        self,
        fk_model: ForwardKinematics,
        dt=None,
        gravity=None,
        noise_params=None,
        epsilon=1e-8,
    ):
        self.g = np.array([0, 0, -9.81]) if gravity is None else np.asarray(gravity)
        self.noise = noise_params if noise_params is not None else NoiseParams()
        self.dt = dt if dt is not None else 0.02  # Default Mujoco time delay
        self.epsilon = epsilon
        self.fk_model = fk_model

        # Constant matrices that is time invariant
        self.A = self._build_A()
        self.Q_k = self._build_cov()
        self.Phi = self._build_phi()
        self.H = self._build_H()

    def _predict(self, state: RobotState, imu_data: IMUMeasurement) -> tuple[np.array]:
        # Easier naming conventions
        X = state.X
        P = state.P

        # Obtain the data from IMU sensors
        omega = imu_data.gyro
        a = imu_data.accel

        # Compute the state propogation
        R, v, p, dl, dr = state.unpack_state()
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

        return state.make_state(R_new, v_new, p_new, dl, dr), P_new

    def _correction(
        self,
        X: np.array,
        P: np.array,
        joint_data: np.array,
    ) -> RobotState:
        # Easier naming convention
        R = X[:3, :3]
        p = X[:3, 4]
        dl = X[:3, 5]
        dr = X[:3, 6]

        # Compute the foot positions and jacobians
        left_pos, J_l, right_pos, J_r = self.fk_model.compute_foot_positions(joint_data)

        # Compute N_t (For both legs) ~ For computation we assume the joint-encoder have standard gaussian noise
        N = self._compute_N(R, J_l, J_r)

        # Some compute optimization to reduce runtime
        PH = P @ self.H.T

        # Compute S_t
        S = self.H @ PH + N

        # Compute Kalman Gain
        K = PH @ np.linalg.inv(S)

        # Predicted foot positions
        dl_new = R @ left_pos + p
        dr_new = R @ right_pos + p

        # Compute the "residual"
        delta = self._compute_delta(dl_new, dr_new, dl, dr)

        # Compute the corrected state
        X_plus = sek3_exp(K @ delta, k=4) @ X

        # Compute the corrected P
        I_KH = np.eye(15) - K @ self.H
        P_plus = I_KH @ P @ I_KH.T + K @ N @ K.T

        return RobotState(X_plus, P_plus)

    def predict(
        self, state: RobotState, imu_data: IMUMeasurement, joint_data: np.array
    ) -> RobotState:
        X_hat, P_hat = self._predict(state, imu_data)
        state = self._correction(X_hat, P_hat, joint_data)

        return state

    def _compute_delta(self, dl_new, dr_new, dl, dr):
        delta = np.zeros((6, 1))
        delta[0:3, :] = (dl_new - dl).reshape(-1, 1)
        delta[3:6, :] = (dr_new - dr).reshape(-1, 1)

        return delta

    @staticmethod
    def _compute_N(R, J_l, J_r):
        N = np.zeros((6, 6))
        rotate_Jl = R @ J_l
        rotate_Jr = R @ J_r
        # We compute our stuff in radians so convert this from degrees to radians
        N[0:3, 0:3] = rotate_Jl @ rotate_Jl.T
        N[3:6, 3:6] = rotate_Jr @ rotate_Jr.T

        return N * (np.pi / 180.0) ** 2

    @staticmethod
    def _build_H():
        H = np.zeros((6, 15))
        # Getting the Body Position
        H[0:3, 6:9] = -np.eye(3)
        H[3:6, 6:9] = -np.eye(3)
        # Left Foot Innovation
        H[0:3, 9:12] = np.eye(3)
        # Right Foot Innovation
        H[3:6, 12:15] = np.eye(3)

        return H

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
        phi[6:9, 0:3] = 0.5 * skew(self.g) * self.dt**2
        phi[6:9, 3:6] = np.eye(3) * self.dt

        return phi
