import numpy as np
from dataclasses import dataclass, field


@dataclass
class IMUMeasurement:
    gyro: np.ndarray  # (3,) angular velocity [rad/s]
    accel: np.ndarray  # (3,) linear acceleration [m/s^2]


@dataclass
class NoiseParams:
    gyro_cov: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01**2)
    accel_cov: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.1**2)
    contact_cov: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.1**2)


# Robot State with no bias augmentation
@dataclass
class RobotState:
    X: np.ndarray = field(default_factory=lambda: np.eye(7))
    P: np.ndarray = field(default_factory=lambda: np.eye(15))

    @staticmethod
    def make_state(R=None, v=None, p=None, dl=None, dr=None):
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

    def unpack_state(self):
        R = self.X[:3, :3]
        v = self.X[:3, 3]
        p = self.X[:3, 4]
        dl = self.X[:3, 5]
        dr = self.X[:3, 6]

        return R, v, p, dl, dr
