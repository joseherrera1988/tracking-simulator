"""Tests for the Track wrapper.

Run with pytest:   pytest test_track.py
Or just directly:  python test_track.py

test_kalman.py covers the filter math. These tests cover the layer on top of it,
and the first one is the one that matters: a Track fed the same measurements as a
bare filter must produce the same estimates. That is what makes this refactor a
refactor. It's a stronger check than watching main.py's printed RMSE, because it
doesn't depend on the demo's seed or its particular numbers.
"""

import numpy as np

from kalman import KalmanFilter2D
from track import Track


def _measurements(n=20, seed=7):
    """A noisy straight-line measurement sequence, shared by the tests below."""
    from sensor import measure

    rng = np.random.default_rng(seed)
    true_xy = np.array([[2.0 * k, 1.0 * k] for k in range(n)])
    return measure(true_xy, meas_std=4.0, rng=rng)


def test_track_matches_bare_filter():
    """A Track is a wrapper, not a change in behavior: same measurements in,
    same estimates out as driving the filter directly."""
    measurements = _measurements()
    dt, process_var, meas_var = 1.0, 0.05, 16.0

    kf = KalmanFilter2D(dt=dt, process_var=process_var, meas_var=meas_var)
    kf.initialize(measurements[0])
    filter_positions = [kf.position]
    for z in measurements[1:]:
        kf.predict()
        kf.update(z)
        filter_positions.append(kf.position)

    track = Track(measurements[0], track_id=1, dt=dt,
                  process_var=process_var, meas_var=meas_var)
    track_positions = [track.position]
    for z in measurements[1:]:
        track_positions.append(track.step(z))

    assert np.allclose(filter_positions, track_positions), (
        "Track estimates diverged from the bare filter's"
    )


def test_coasting_follows_the_prediction():
    """With no measurement, the track advances on its velocity alone and counts
    the miss. This path is unused by the single-target demo -- gating and
    association need it."""
    measurements = _measurements()
    track = Track(measurements[0], track_id=1, dt=1.0, process_var=0.05,
                  meas_var=16.0)
    for z in measurements[1:10]:
        track.step(z)

    position_before = track.position
    velocity = track.velocity
    coasted = track.step(None)

    expected = position_before + velocity  # dt = 1.0
    assert np.allclose(coasted, expected), (
        f"coasted to {coasted}, expected {expected}"
    )
    assert track.misses == 1, f"expected 1 miss, got {track.misses}"


def test_counters_track_hits_and_misses():
    """hits counts the measurement that created the track, so a fresh track
    starts at 1. A hit clears any accumulated misses."""
    measurements = _measurements()
    track = Track(measurements[0], track_id=1, dt=1.0, process_var=0.05,
                  meas_var=16.0)

    assert track.hits == 1, "the initializing measurement counts as a hit"
    assert track.misses == 0
    assert track.status == "tentative", "a new track has to earn confirmation"

    track.step(measurements[1])
    assert track.hits == 2, f"expected 2 hits, got {track.hits}"

    track.step(None)
    track.step(None)
    assert track.misses == 2, f"expected 2 misses, got {track.misses}"

    track.step(measurements[2])
    assert track.misses == 0, "a hit should reset the miss counter"
    assert track.hits == 3, f"expected 3 hits, got {track.hits}"


def test_track_keeps_its_id():
    """IDs come from the caller; the track just holds onto one."""
    measurements = _measurements()
    track = Track(measurements[0], track_id=42, dt=1.0, process_var=0.05,
                  meas_var=16.0)
    track.step(measurements[1])
    assert track.id == 42


def test_split_predict_correct_matches_step():
    """step() is exactly predict() then correct(), so gating can slot into the
    gap without changing what the track does."""
    measurements = _measurements()
    kwargs = dict(dt=1.0, process_var=0.05, meas_var=16.0)

    fused = Track(measurements[0], track_id=1, **kwargs)
    split = Track(measurements[0], track_id=2, **kwargs)

    for z in measurements[1:]:
        fused.step(z)
        split.predict()
        split.correct(z)

    assert np.allclose(fused.position, split.position), (
        "splitting the scan changed the estimate"
    )
    assert fused.hits == split.hits


def test_gating_sees_the_predicted_state_not_the_updated_one():
    """The whole reason predict and correct are separate: between them the
    track reports where it EXPECTS a measurement, which is what the gate is
    judged against. After correct() that value has moved."""
    measurements = _measurements()
    track = Track(measurements[0], track_id=1, dt=1.0, process_var=0.05,
                  meas_var=16.0)
    track.step(measurements[1])

    track.predict()
    predicted = track.predicted_measurement.copy()
    track.correct(measurements[2])

    assert not np.allclose(predicted, track.predicted_measurement), (
        "reading the prediction after correct() should not give the pre-update "
        "value -- if these match, the gate is being judged against the wrong state"
    )


def test_track_exposes_prediction_quantities():
    """Gating talks to the Track, not to the filter inside it."""
    measurements = _measurements()
    track = Track(measurements[0], track_id=1, dt=1.0, process_var=0.05,
                  meas_var=16.0)
    track.step(measurements[1])

    assert np.allclose(track.predicted_measurement,
                       track.kf.predicted_measurement)
    assert np.allclose(track.innovation_covariance,
                       track.kf.innovation_covariance)


if __name__ == "__main__":
    # lets you run the file directly without pytest installed
    test_track_matches_bare_filter()
    test_coasting_follows_the_prediction()
    test_counters_track_hits_and_misses()
    test_track_keeps_its_id()
    test_split_predict_correct_matches_step()
    test_gating_sees_the_predicted_state_not_the_updated_one()
    test_track_exposes_prediction_quantities()
    print("all tests passed")
