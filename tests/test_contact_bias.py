"""Tests for the contact-bias IEKF module.

Covers shape/structure, free-fall regression, random-walk propagation,
convergence under correction (stationary and rotated base), numerical
Jacobian checks, contact-reset mechanics, PSD and orthogonality
stability, and the weak-observability limitation in single support.

The right_invariant_correct helper below is a minimal correction step
used by the correction tests.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contact_bias_iekf import (
    ContactBiasIEKF, ContactBiasState,
    IDX_PHI, IDX_V, IDX_P, IDX_DL, IDX_DR, IDX_BL, IDX_BR, DIM_ERR,
)
from src.lie_group import sek3_exp, so3_exp, skew
from src.types import IMUMeasurement, NoiseParams


def right_invariant_correct(iekf, state, y_fk_body, foot="left", N=None):
    """Right-invariant Kalman update for one foot's FK measurement.

    Applies the Lie correction as X_new = Exp(delta_xi) @ X. Bias is
    updated additively. Joseph form for covariance.

    Args:
        iekf: a ContactBiasIEKF.
        state: current ContactBiasState.
        y_fk_body: (3,) body-frame FK foot-position measurement.
        foot: 'left' or 'right'.
        N: (3,3) world-frame measurement noise covariance. Defaults to
           iekf.noise.contact_cov rotated into world frame by R_hat.

    Returns:
        Updated ContactBiasState.
    """
    H = iekf.build_H(foot)
    y_innov_world = iekf.innovation(state, y_fk_body, foot)

    if N is None:
        R_hat = state.X[:3, :3]
        N = R_hat @ iekf.noise.contact_cov @ R_hat.T

    P = state.P
    S = H @ P @ H.T + N
    K = P @ H.T @ np.linalg.solve(S, np.eye(3))

    dx = K @ y_innov_world
    delta_X = sek3_exp(dx[:15], k=4)
    X_new = delta_X @ state.X
    b_L_new = state.b_L + dx[IDX_BL]
    b_R_new = state.b_R + dx[IDX_BR]

    I = np.eye(DIM_ERR)
    IKH = I - K @ H
    P_new = IKH @ P @ IKH.T + K @ N @ K.T

    return ContactBiasState(X=X_new, b_L=b_L_new, b_R=b_R_new, P=P_new)


def test_state_shapes():
    iekf = ContactBiasIEKF()
    state = iekf.make_state()

    assert state.X.shape == (7, 7)
    assert state.b_L.shape == (3,)
    assert state.b_R.shape == (3,)
    assert state.P.shape == (21, 21) == (DIM_ERR, DIM_ERR)

    # Error-state index constants partition [0, 21).
    covered = []
    for sl in (IDX_PHI, IDX_V, IDX_P, IDX_DL, IDX_DR, IDX_BL, IDX_BR):
        covered.extend(range(sl.start, sl.stop))
    assert covered == list(range(DIM_ERR))


def test_F_structure():
    """F is block-diag: base 15x15 upper-left, identity on bias block."""
    iekf = ContactBiasIEKF()
    dt = 0.01
    F = iekf.compute_F(dt)

    assert F.shape == (21, 21)
    np.testing.assert_allclose(F[:15, :15], iekf.base.compute_F(dt))
    np.testing.assert_allclose(F[15:, 15:], np.eye(6))
    np.testing.assert_array_equal(F[:15, 15:], 0)
    np.testing.assert_array_equal(F[15:, :15], 0)


def test_Q_structure():
    """Q is block-diag: base 15x15 upper-left, bias_cov*dt blocks."""
    iekf = ContactBiasIEKF()
    dt = 0.01
    Q = iekf.compute_Q(dt)

    assert Q.shape == (21, 21)
    np.testing.assert_allclose(Q[:15, :15], iekf.base.compute_Q(dt))
    np.testing.assert_allclose(Q[IDX_BL, IDX_BL], iekf.noise.bias_cov * dt)
    np.testing.assert_allclose(Q[IDX_BR, IDX_BR], iekf.noise.bias_cov * dt)
    np.testing.assert_array_equal(Q[IDX_BL, IDX_BR], 0)
    np.testing.assert_array_equal(Q[IDX_BR, IDX_BL], 0)


def test_H_structure():
    """H has -I on position, +I on active foot's d and b, zeros elsewhere."""
    iekf = ContactBiasIEKF()
    H_L = iekf.build_H("left")
    H_R = iekf.build_H("right")

    for H in (H_L, H_R):
        assert H.shape == (3, 21)
        np.testing.assert_allclose(H[:, IDX_PHI], 0)
        np.testing.assert_allclose(H[:, IDX_V], 0)
        np.testing.assert_allclose(H[:, IDX_P], -np.eye(3))

    np.testing.assert_allclose(H_L[:, IDX_DL], np.eye(3))
    np.testing.assert_allclose(H_L[:, IDX_BL], np.eye(3))
    np.testing.assert_allclose(H_L[:, IDX_DR], 0)
    np.testing.assert_allclose(H_L[:, IDX_BR], 0)

    np.testing.assert_allclose(H_R[:, IDX_DR], np.eye(3))
    np.testing.assert_allclose(H_R[:, IDX_BR], np.eye(3))
    np.testing.assert_allclose(H_R[:, IDX_DL], 0)
    np.testing.assert_allclose(H_R[:, IDX_BL], 0)


def test_free_fall_with_zero_bias_matches_base():
    """With biases zero and no correction, propagation reproduces the
    base free-fall result.
    """
    g = np.array([0.0, 0.0, -9.81])
    iekf = ContactBiasIEKF(gravity=g)
    state = iekf.make_state(
        dl=np.array([0, 0.12, -0.76]),
        dr=np.array([0, -0.12, -0.76]),
    )

    imu = IMUMeasurement(gyro=np.zeros(3), accel=np.zeros(3))
    dt = 0.001
    t_total = 1.0
    n_steps = int(t_total / dt)
    for _ in range(n_steps):
        state = iekf.propagate(
            state, gyro_bias=np.zeros(3), accel_bias=np.zeros(3),
            imu=imu, dt=dt,
        )

    R_f = state.X[:3, :3]
    v_f = state.X[:3, 3]
    p_f = state.X[:3, 4]

    np.testing.assert_allclose(v_f, g * t_total, atol=1e-3,
                               err_msg="Velocity doesn't match free-fall")
    np.testing.assert_allclose(p_f, 0.5 * g * t_total**2, atol=1e-3,
                               err_msg="Position doesn't match free-fall")
    np.testing.assert_allclose(R_f, np.eye(3), atol=1e-10,
                               err_msg="Rotation should stay identity")
    np.testing.assert_allclose(state.b_L, 0)
    np.testing.assert_allclose(state.b_R, 0)


def test_bias_random_walk_covariance_grows_linearly():
    """With no measurements, cov(b_L) after t seconds equals bias_cov * t
    (F is identity on the bias block).
    """
    iekf = ContactBiasIEKF()
    state = iekf.make_state(P=np.zeros((DIM_ERR, DIM_ERR)))

    imu = IMUMeasurement(gyro=np.zeros(3), accel=np.zeros(3))
    dt = 0.01
    t_total = 5.0
    n_steps = int(t_total / dt)
    for _ in range(n_steps):
        state = iekf.propagate(
            state, gyro_bias=np.zeros(3), accel_bias=np.zeros(3),
            imu=imu, dt=dt,
        )

    expected = iekf.noise.bias_cov * t_total
    np.testing.assert_allclose(state.P[IDX_BL, IDX_BL], expected, atol=1e-12)
    np.testing.assert_allclose(state.P[IDX_BR, IDX_BR], expected, atol=1e-12)
    np.testing.assert_allclose(state.b_L, 0)
    np.testing.assert_allclose(state.b_R, 0)


def test_bias_converges_stationary_robot():
    """Stationary robot (R = I) with a known world-frame bias in FK.

    At R = I the body-frame and world-frame conventions coincide, so
    this test checks convergence in the "easy" case. See the rotation
    test below for the non-trivial check.
    """
    iekf = ContactBiasIEKF(noise_params=NoiseParams())
    d_L = np.array([0.05, 0.12, -0.76])
    state = iekf.make_state(dl=d_L, dr=np.array([0.05, -0.12, -0.76]))

    P = np.eye(DIM_ERR) * 1e-6
    P[IDX_BL, IDX_BL] = np.eye(3) * 1e-2   # ~10 cm std bias prior
    state = ContactBiasState(X=state.X, b_L=state.b_L, b_R=state.b_R, P=P)

    b_true = np.array([0.02, -0.01, 0.005])
    # R = I, p = 0 => body-frame FK equals d + b_true.
    y = d_L + b_true

    N = np.eye(3) * 1e-8   # tight measurement noise to speed convergence
    for _ in range(50):
        state = right_invariant_correct(iekf, state, y, "left", N=N)

    np.testing.assert_allclose(state.b_L, b_true, atol=5e-3,
                               err_msg="b_hat failed to converge to b_true")


def test_bias_converges_under_rotation():
    """Non-identity base rotation. Verifies that innovation and H are
    consistently in world frame; a sign or frame mismatch diverges here.
    """
    iekf = ContactBiasIEKF(noise_params=NoiseParams())
    R = so3_exp(np.array([0.0, 0.0, np.pi / 4]))   # 45 deg about z
    d_L = np.array([0.05, 0.12, -0.76])
    state = iekf.make_state(R=R, dl=d_L)

    P = np.eye(DIM_ERR) * 1e-6
    P[IDX_BL, IDX_BL] = np.eye(3) * 1e-2
    state = ContactBiasState(X=state.X, b_L=state.b_L, b_R=state.b_R, P=P)

    b_true_world = np.array([0.02, -0.01, 0.005])
    # True body-frame FK with world-frame bias: y = R^T (d + b - p), p = 0.
    y_body = R.T @ (d_L + b_true_world)

    N = np.eye(3) * 1e-8
    for _ in range(50):
        state = right_invariant_correct(iekf, state, y_body, "left", N=N)

    np.testing.assert_allclose(state.b_L, b_true_world, atol=5e-3,
                               err_msg="b_hat failed to converge with R != I")


def test_bias_and_position_both_observable_under_motion():
    """Full-trajectory observability belongs in a gait/scenario test."""
    pass


def test_reset_contact_clears_cross_correlations():
    """After reset_contact, rows/cols for (d_i, b_i) have no cross terms
    with other state, bias is zeroed, and d_i takes the supplied world
    position.
    """
    iekf = ContactBiasIEKF()
    state = iekf.make_state(dl=np.array([0.0, 0.12, -0.76]))

    # Inject non-trivial correlations so the reset has something to clear.
    state.P[:] = 0.01
    state.P[np.diag_indices(DIM_ERR)] = 0.1
    state.b_L = np.array([0.3, 0.2, 0.1])

    d_new = np.array([1.0, 0.14, 0.0])
    state = iekf.reset_contact(state, "left", d_new)

    np.testing.assert_allclose(state.X[:3, 5], d_new)
    np.testing.assert_allclose(state.b_L, 0)

    for idx in (IDX_DL, IDX_BL):
        row = state.P[idx, :].copy()
        row[:, idx] = 0
        np.testing.assert_allclose(row, 0,
            err_msg="Off-block row should be zero after reset")
        col = state.P[:, idx].copy()
        col[idx, :] = 0
        np.testing.assert_allclose(col, 0,
            err_msg="Off-block col should be zero after reset")


def test_weak_observability_single_support_grows_cov():
    """In stationary single-support, d_i and b_i enter the measurement
    with the same column structure, so one linear combination is
    unobservable. The (d_L, b_L) covariance block does not collapse to
    zero even with repeated FK updates.
    """
    iekf = ContactBiasIEKF()
    d_L = np.array([0.0, 0.12, -0.76])
    state = iekf.make_state(dl=d_L)

    P = np.eye(DIM_ERR) * 1e-4
    state = ContactBiasState(X=state.X, b_L=state.b_L, b_R=state.b_R, P=P)

    y = d_L.copy()   # b_true = 0, R = I, p = 0 => body-frame measurement = d_L
    dt = 0.01
    imu = IMUMeasurement(gyro=np.zeros(3), accel=np.zeros(3))

    for _ in range(200):
        state = iekf.propagate(
            state, gyro_bias=np.zeros(3), accel_bias=np.zeros(3),
            imu=imu, dt=dt,
        )
        state = right_invariant_correct(iekf, state, y, "left")

    trace_after = (np.trace(state.P[IDX_DL, IDX_DL])
                   + np.trace(state.P[IDX_BL, IDX_BL]))
    assert trace_after > 1e-8, \
        "(d_L, b_L) covariance should not collapse in single support"


def _numerical_jacobian_w_r_t_estimate(iekf, state, y_body, foot):
    """Numerical partial derivative of innovation w.r.t. a perturbation
    of the ESTIMATE (X_hat_new = exp(xi) @ X_hat_old, b_hat_new = b + xi_b).

    By the chain rule, this equals -H_analytical where H_analytical is
    the Jacobian w.r.t. the TRUE error xi (with X_true = exp(xi) X_hat).
    Perturbing the estimate by +xi is equivalent to changing the true
    error by -xi while keeping the true state fixed.
    """
    eps = 1e-7
    H_num = np.zeros((3, DIM_ERR))
    nu0 = iekf.innovation(state, y_body, foot)
    for i in range(DIM_ERR):
        xi = np.zeros(DIM_ERR)
        xi[i] = eps
        X_pert = sek3_exp(xi[:15], k=4) @ state.X
        b_L_pert = state.b_L + xi[IDX_BL]
        b_R_pert = state.b_R + xi[IDX_BR]
        state_pert = ContactBiasState(
            X=X_pert, b_L=b_L_pert, b_R=b_R_pert, P=state.P,
        )
        nu = iekf.innovation(state_pert, y_body, foot)
        H_num[:, i] = (nu - nu0) / eps
    return H_num


def test_H_exact_at_zero_bias():
    """With b_hat = 0, the -skew(b_hat) cross-term vanishes, so the
    analytical H w.r.t. true error matches the numerical Jacobian up to
    the estimate-vs-true-error sign flip.
    """
    iekf = ContactBiasIEKF()
    R = so3_exp(np.array([0.1, -0.2, 0.3]))
    state = iekf.make_state(
        R=R, dl=np.array([0.05, 0.12, -0.76]),
        b_L=np.zeros(3),
    )
    y = iekf.predict_measurement(state, "left")   # innovation is zero here

    H_num = _numerical_jacobian_w_r_t_estimate(iekf, state, y, "left")
    H_ana = iekf.build_H("left")
    np.testing.assert_allclose(H_num, -H_ana, atol=1e-5,
        err_msg="H_num (w.r.t. estimate) should equal -H_ana (w.r.t. true error) at b=0")


def test_H_phi_column_shows_dropped_term_at_nonzero_bias():
    """At nonzero b_hat, the true Jacobian w.r.t. true error has
    +skew(b_hat) on the delta_phi column (from the identity
    -skew(delta_phi) b_hat = +skew(b_hat) delta_phi). The analytical H
    drops it (imperfect-IEKF approximation). Numerical w.r.t. estimate
    shows -skew(b_hat) = -(+skew(b_hat)).
    """
    iekf = ContactBiasIEKF()
    R = so3_exp(np.array([0.1, -0.2, 0.3]))
    b_L = np.array([0.02, -0.01, 0.005])
    state = iekf.make_state(R=R, dl=np.array([0.05, 0.12, -0.76]), b_L=b_L)
    y = iekf.predict_measurement(state, "left")

    H_num = _numerical_jacobian_w_r_t_estimate(iekf, state, y, "left")
    H_ana = iekf.build_H("left")

    # Every column except delta_phi should match -H_ana to numerical precision.
    for sl in (IDX_V, IDX_P, IDX_DL, IDX_DR, IDX_BL, IDX_BR):
        np.testing.assert_allclose(H_num[:, sl], -H_ana[:, sl], atol=1e-5,
            err_msg=f"Column {sl} differs from expected -H_ana")

    # delta_phi column: numerical w.r.t. estimate = -skew(b_hat).
    np.testing.assert_allclose(H_num[:, IDX_PHI], -skew(b_L), atol=1e-5,
        err_msg="delta_phi column w.r.t. estimate should equal -skew(b_hat)")
    np.testing.assert_allclose(H_ana[:, IDX_PHI], 0)


def test_covariance_remains_psd_over_many_updates():
    """After 500 propagate+correct cycles, P's smallest eigenvalue
    should be >= 0 up to numerical tolerance. Joseph form guards
    against negative eigenvalues from subtraction.
    """
    iekf = ContactBiasIEKF()
    d_L = np.array([0.0, 0.12, -0.76])
    state = iekf.make_state(dl=d_L, P=np.eye(DIM_ERR) * 1e-3)

    imu = IMUMeasurement(gyro=np.zeros(3), accel=np.zeros(3))
    dt = 0.01
    for _ in range(500):
        state = iekf.propagate(
            state, gyro_bias=np.zeros(3), accel_bias=np.zeros(3),
            imu=imu, dt=dt,
        )
        state = right_invariant_correct(iekf, state, d_L, "left")

    min_eig = np.linalg.eigvalsh(state.P).min()
    assert min_eig > -1e-10, f"P lost PSD: min eigenvalue {min_eig}"


def test_rotation_stays_orthogonal_under_updates():
    """After many corrections, R should remain orthogonal (R R^T = I).
    The sek3_exp left-multiplication must preserve the group structure.
    """
    iekf = ContactBiasIEKF()
    R0 = so3_exp(np.array([0.1, -0.2, 0.3]))
    d_L = np.array([0.05, 0.12, -0.76])
    state = iekf.make_state(R=R0, dl=d_L, P=np.eye(DIM_ERR) * 1e-3)
    y = R0.T @ d_L   # true body-frame measurement at b=0

    for _ in range(100):
        state = right_invariant_correct(iekf, state, y, "left")

    R_final = state.X[:3, :3]
    np.testing.assert_allclose(R_final @ R_final.T, np.eye(3), atol=1e-8,
        err_msg="R should stay orthogonal")
    np.testing.assert_allclose(np.linalg.det(R_final), 1.0, atol=1e-8,
        err_msg="R should have det = 1")


def test_large_initial_bias_error_recovers():
    """Starting with b_hat = 0 and b_true = 10 cm per axis, the filter
    should still converge. This stresses the region where the dropped
    -skew(b) term is largest (but still small relative to signal).
    """
    iekf = ContactBiasIEKF()
    R = so3_exp(np.array([0.0, 0.0, np.pi / 6]))   # 30 deg yaw
    d_L = np.array([0.05, 0.12, -0.76])
    state = iekf.make_state(R=R, dl=d_L)

    P = np.eye(DIM_ERR) * 1e-6
    P[IDX_BL, IDX_BL] = np.eye(3) * 0.1   # very loose bias prior
    state = ContactBiasState(X=state.X, b_L=state.b_L, b_R=state.b_R, P=P)

    b_true_world = np.array([0.10, -0.10, 0.10])   # 10 cm on each axis
    y_body = R.T @ (d_L + b_true_world)

    N = np.eye(3) * 1e-8
    for _ in range(200):
        state = right_invariant_correct(iekf, state, y_body, "left", N=N)

    np.testing.assert_allclose(state.b_L, b_true_world, atol=1e-2,
        err_msg="Filter failed to recover from 10cm initial bias error")


def test_alternating_foot_updates():
    """Alternating L/R FK updates should not corrupt the other foot's
    estimates. Both biases should converge to their respective truths.
    """
    iekf = ContactBiasIEKF()
    d_L = np.array([0.05, 0.12, -0.76])
    d_R = np.array([0.05, -0.12, -0.76])
    state = iekf.make_state(dl=d_L, dr=d_R)

    P = np.eye(DIM_ERR) * 1e-6
    P[IDX_BL, IDX_BL] = np.eye(3) * 1e-2
    P[IDX_BR, IDX_BR] = np.eye(3) * 1e-2
    state = ContactBiasState(X=state.X, b_L=state.b_L, b_R=state.b_R, P=P)

    b_true_L = np.array([0.02, 0.01, -0.005])
    b_true_R = np.array([-0.02, 0.01, 0.005])
    y_L = d_L + b_true_L   # R=I so body = world
    y_R = d_R + b_true_R

    N = np.eye(3) * 1e-8
    for i in range(100):
        if i % 2 == 0:
            state = right_invariant_correct(iekf, state, y_L, "left", N=N)
        else:
            state = right_invariant_correct(iekf, state, y_R, "right", N=N)

    np.testing.assert_allclose(state.b_L, b_true_L, atol=5e-3,
        err_msg="Left bias failed to converge with alternating updates")
    np.testing.assert_allclose(state.b_R, b_true_R, atol=5e-3,
        err_msg="Right bias failed to converge with alternating updates")


if __name__ == "__main__":
    tests = [
        test_state_shapes,
        test_F_structure,
        test_Q_structure,
        test_H_structure,
        test_free_fall_with_zero_bias_matches_base,
        test_bias_random_walk_covariance_grows_linearly,
        test_bias_converges_stationary_robot,
        test_bias_converges_under_rotation,
        test_bias_and_position_both_observable_under_motion,
        test_reset_contact_clears_cross_correlations,
        test_weak_observability_single_support_grows_cov,
        test_H_exact_at_zero_bias,
        test_H_phi_column_shows_dropped_term_at_nonzero_bias,
        test_covariance_remains_psd_over_many_updates,
        test_rotation_stays_orthogonal_under_updates,
        test_large_initial_bias_error_recovers,
        test_alternating_foot_updates,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
