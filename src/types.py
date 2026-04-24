import numpy as np
from dataclasses import dataclass, field


@dataclass
class IMUMeasurement:
    gyro: np.ndarray      # (3,) angular velocity [rad/s]
    accel: np.ndarray     # (3,) linear acceleration [m/s^2]
    timestamp: float = 0.0


@dataclass
class NoiseParams:
    gyro_cov: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 0.01
    )
    accel_cov: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 0.1
    )
    contact_cov: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 0.01
    )
    # Per-foot contact-position bias random-walk process noise (body frame).
    # sigma_b ~ 1e-3 m/sqrt(s) per axis -> (1e-3)^2 = 1e-6.
    bias_cov: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 1e-6
    )