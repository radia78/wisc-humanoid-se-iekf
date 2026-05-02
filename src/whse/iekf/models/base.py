import numpy as np
from whse.iekf.dynamics import RIEKFDynamics
from whse.iekf.types import IMUMeasurement, RobotState
from whse.iekf.lie_group import sek3_exp

from whse.kinematics import ForwardKinematics


class RIEKF:
    def __init__(
        self,
        fk_model: ForwardKinematics,
        dt=None,
        gravity=None,
        noise_params=None,
        epsilon=1e-8,
    ):
        self._base = RIEKFDynamics(dt=dt, gravity=gravity, noise_params=noise_params)
        self.epsilon = epsilon
        self.fk_model = fk_model
        self.H = self._build_H()

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

    def predict(
        self,
        state: RobotState,
        imu_data: IMUMeasurement,
        joint_data: np.array,
    ):
        # Predict the current state based on previous state
        X, P = self._base.propagate(state, imu_data)

        # Easier readability
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
        N[0:3, 0:3] = rotate_Jl @ rotate_Jl.T * 0.01745329
        N[3:6, 3:6] = rotate_Jr @ rotate_Jr.T * 0.01745329

        return N

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
