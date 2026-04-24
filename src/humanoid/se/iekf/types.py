import numpy as np
from dataclasses import dataclass, field


@dataclass
class IMUMeasurement:
    gyro: np.ndarray  # (3,) angular velocity [rad/s]
    accel: np.ndarray  # (3,) linear acceleration [m/s^2]
    timestamp: float = 0.0


@dataclass
class NoiseParams:
    gyro_cov: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
    accel_cov: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.1)
    contact_cov: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
