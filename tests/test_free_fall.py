"""Free-fall integration test.

Drops the robot from rest with no rotation. After t seconds of free fall:
  v = g * t
  p = 0.5 * g * t^2

Verifies that dynamics propagation matches analytical free-fall trajectory.
"""

import numpy as np

from humanoid.se.iekf.dynamics import IEKFDynamics
from humanoid.se.iekf.types import IMUMeasurement, NoiseParams


def test_free_fall():
    g = np.array([0.0, 0.0, -9.81])
    dynamics = IEKFDynamics(gravity=g, noise_params=NoiseParams())

    # Start at rest, identity rotation, origin position
    X = dynamics.make_state(
        R=np.eye(3),
        v=np.zeros(3),
        p=np.zeros(3),
        dl=np.array([0, 0.12, -0.76]),
        dr=np.array([0, -0.12, -0.76]),
    )

    gyro_bias = np.zeros(3)
    accel_bias = np.zeros(3)

    dt = 0.001  # 1 kHz
    t_total = 1.0  # 1 second of free fall
    n_steps = int(t_total / dt)

    # Free-fall IMU: accelerometer reads zero (only gravity, no contact force)
    imu = IMUMeasurement(
        gyro=np.zeros(3),
        accel=np.zeros(3),  # free fall → accel = 0 (gravity not sensed)
    )

    for _ in range(n_steps):
        X = dynamics.propagate_state(X, gyro_bias, accel_bias, imu, dt)

    R, v, p, dl, dr = dynamics.unpack_state(X)

    # Analytical: v = g*t, p = 0.5*g*t^2
    v_expected = g * t_total
    p_expected = 0.5 * g * t_total**2

    print(f"v final:    {v}")
    print(f"v expected: {v_expected}")
    print(f"p final:    {p}")
    print(f"p expected: {p_expected}")

    np.testing.assert_allclose(
        v, v_expected, atol=1e-3, err_msg="Velocity doesn't match free-fall"
    )
    np.testing.assert_allclose(
        p, p_expected, atol=1e-3, err_msg="Position doesn't match free-fall"
    )
    np.testing.assert_allclose(
        R, np.eye(3), atol=1e-10, err_msg="Rotation should stay identity"
    )

    print("\n[PASS] Free-fall test passed!")


if __name__ == "__main__":
    test_free_fall()
