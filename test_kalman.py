"""Tests for the Kalman filter.

Run with pytest:   pytest test_kalman.py
Or just directly:  python test_kalman.py

The important test here is a "known-answer" test: we hand the filter perfect,
noise-free measurements from a target whose true motion we know exactly, then
check that the filter recovers that motion. If the filter can't track a target
when there's NO noise to see through, something is wrong with the core math --
so this catches the kind of bug (a wrong sign, a transposed matrix) that a plot
might let you miss.
"""

import numpy as np

from kalman import KalmanFilter2D


def test_recovers_position_and_velocity_on_clean_track():
    """With noise-free measurements, the estimate should converge onto the true
    position AND recover the true velocity -- even though velocity is never
    measured, only inferred from how position changes."""
    dt = 1.0
    true_vx, true_vy = 3.0, -2.0  # the velocity the filter must discover

    # true positions along a perfectly straight line, no noise
    n = 30
    true_xy = np.array([[true_vx * k * dt, true_vy * k * dt] for k in range(n)])

    kf = KalmanFilter2D(dt=dt, process_var=0.01, meas_var=1.0)
    kf.initialize(true_xy[0])
    for k in range(1, n):
        kf.predict()
        kf.update(true_xy[k])

    # after settling, position estimate should sit essentially on truth
    pos_error = np.linalg.norm(kf.position - true_xy[-1])
    assert pos_error < 0.5, f"position error too large: {pos_error}"

    # and the inferred velocity should match the true velocity
    vel_error = np.linalg.norm(kf.velocity - np.array([true_vx, true_vy]))
    assert vel_error < 0.1, f"velocity error too large: {vel_error}"


def test_filter_beats_raw_measurements_with_noise():
    """A sanity check on the whole point of the filter: with noisy measurements,
    the filtered track should have clearly lower error than the raw blips."""
    from sensor import measure
    from metrics import position_rmse

    rng = np.random.default_rng(0)
    dt, meas_std, n = 1.0, 5.0, 50
    true_xy = np.array([[2.0 * k, 1.0 * k] for k in range(n)])
    measurements = measure(true_xy, meas_std=meas_std, rng=rng)

    kf = KalmanFilter2D(dt=dt, process_var=0.05, meas_var=meas_std**2)
    kf.initialize(measurements[0])
    estimates = [kf.position]
    for k in range(1, n):
        kf.predict()
        kf.update(measurements[k])
        estimates.append(kf.position)

    track_rmse = position_rmse(estimates, true_xy)
    raw_rmse = position_rmse(measurements, true_xy)
    assert track_rmse < raw_rmse, (
        f"filter ({track_rmse:.2f}) should beat raw ({raw_rmse:.2f})"
    )


def test_prediction_quantities_available_before_any_update():
    """Gating needs the predicted measurement and the innovation covariance
    between predict() and update() -- before a measurement has been chosen.
    Checked against hand-computed values, since an index slip in H or P would
    still produce a plausible-looking 2x2."""
    meas_var = 16.0
    kf = KalmanFilter2D(dt=1.0, process_var=0.05, meas_var=meas_var)
    kf.initialize([10.0, -4.0])

    # the predicted measurement is the position part of the state
    assert np.allclose(kf.predicted_measurement, [10.0, -4.0])
    assert np.allclose(kf.predicted_measurement, kf.position)

    # initialize() sets the position variances to meas_var, so S starts at
    # meas_var (state) + meas_var (sensor) on each axis, with no cross terms.
    S = kf.innovation_covariance
    assert S.shape == (2, 2), f"expected a 2x2, got {S.shape}"
    assert np.allclose(S, np.diag([2 * meas_var, 2 * meas_var])), f"S was {S}"

    # predicting grows the uncertainty, so the expected spread grows with it
    S_before = kf.innovation_covariance
    kf.predict()
    S_after = kf.innovation_covariance
    assert np.all(np.diag(S_after) > np.diag(S_before)), (
        "S should grow across a predict step"
    )
    assert np.allclose(S_after, S_after.T), "S must stay symmetric"


if __name__ == "__main__":
    # lets you run the file directly without pytest installed
    test_recovers_position_and_velocity_on_clean_track()
    test_filter_beats_raw_measurements_with_noise()
    test_prediction_quantities_available_before_any_update()
    print("all tests passed")
