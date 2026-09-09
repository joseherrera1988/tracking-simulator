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


if __name__ == "__main__":
    # lets you run the file directly without pytest installed
    test_recovers_position_and_velocity_on_clean_track()
    test_filter_beats_raw_measurements_with_noise()
    print("all tests passed")
