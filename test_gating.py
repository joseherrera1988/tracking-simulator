"""Tests for the gating distance.

Run with pytest:   pytest test_gating.py
Or just directly:  python test_gating.py

These check against distances worked out by hand rather than against whatever
the code happens to produce. The failure mode being guarded here is subtle: a
transposed S or a dropped inverse still yields plausible-looking positive
numbers, and a gate built on them would not crash -- it would just quietly admit
the wrong measurements and track worse.
"""

import numpy as np

from gating import squared_mahalanobis_distances, CHI2_99_2D


def test_identity_covariance_is_plain_squared_distance():
    """With S = I, the Mahalanobis distance reduces to squared Euclidean
    distance -- the sanity anchor for everything else."""
    z_hat = np.array([0.0, 0.0])
    S = np.eye(2)
    measurements = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])

    d2 = squared_mahalanobis_distances(z_hat, S, measurements)

    assert np.allclose(d2, [25.0, 1.0, 0.0]), f"got {d2}"


def test_covariance_scales_each_axis():
    """S divides by the expected spread per axis: a measurement two std-devs
    out on x and three on y is the same distance as any other 1-sigma step."""
    z_hat = np.array([0.0, 0.0])
    S = np.diag([4.0, 9.0])            # std-devs of 2 and 3

    # (2, 3) is exactly one std-dev out on each axis -> 1 + 1 = 2
    d2 = squared_mahalanobis_distances(z_hat, S, np.array([[2.0, 3.0]]))
    assert np.allclose(d2, [2.0]), f"got {d2}"

    # the same 2-unit offset costs far less on the wider y axis than on x
    on_x = squared_mahalanobis_distances(z_hat, S, np.array([[2.0, 0.0]]))
    on_y = squared_mahalanobis_distances(z_hat, S, np.array([[0.0, 2.0]]))
    assert on_x > on_y, f"x offset {on_x} should cost more than y offset {on_y}"


def test_correlated_covariance():
    """Off-diagonal terms matter -- this is where a transposed or wrongly
    inverted S stops being detectable by symmetric test cases.

    S = [[2, 1], [1, 2]], so S^-1 = (1/3) [[2, -1], [-1, 2]].
    For y = [1, 0]:  d^2 = 2/3.
    For y = [1, 1]:  d^2 = (1/3)(2 - 1 - 1 + 2) = 2/3.
    For y = [1, -1]: d^2 = (1/3)(2 + 1 + 1 + 2) = 2.
    The last two differ because the covariance says x and y err together, so a
    measurement off in the same direction on both axes is less surprising than
    one off in opposite directions.
    """
    z_hat = np.array([0.0, 0.0])
    S = np.array([[2.0, 1.0], [1.0, 2.0]])
    measurements = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])

    d2 = squared_mahalanobis_distances(z_hat, S, measurements)

    assert np.allclose(d2, [2 / 3, 2 / 3, 2.0]), f"got {d2}"


def test_offset_prediction_and_single_measurement():
    """The prediction is subtracted off, and a lone [x, y] is accepted the same
    way sensor.measure() accepts one position."""
    z_hat = np.array([10.0, -5.0])
    S = np.eye(2)

    d2 = squared_mahalanobis_distances(z_hat, S, [13.0, -1.0])

    assert d2.shape == (1,), f"expected one distance, got shape {d2.shape}"
    assert np.allclose(d2, [25.0]), f"got {d2}"


def test_gate_admits_the_true_measurement_and_rejects_a_far_one():
    """End to end against a real track: the measurement the target actually
    produced falls inside the gate, and a point far off does not."""
    from sensor import measure
    from track import Track

    rng = np.random.default_rng(3)
    n = 15
    true_xy = np.array([[2.0 * k, 1.0 * k] for k in range(n)])
    measurements = measure(true_xy, meas_std=4.0, rng=rng)

    track = Track(measurements[0], track_id=1, dt=1.0, process_var=0.05,
                  meas_var=16.0)
    for z in measurements[1:-1]:
        track.step(z)

    # the track has predicted; now judge candidates for the final scan
    track.kf.predict()
    real = measurements[-1]
    decoy = real + np.array([120.0, -90.0])

    d2 = squared_mahalanobis_distances(
        track.predicted_measurement, track.innovation_covariance,
        np.array([real, decoy]),
    )

    assert d2[0] <= CHI2_99_2D, (
        f"the true measurement should gate in, distance was {d2[0]:.2f}"
    )
    assert d2[1] > CHI2_99_2D, (
        f"the far decoy should gate out, distance was {d2[1]:.2f}"
    )


if __name__ == "__main__":
    # lets you run the file directly without pytest installed
    test_identity_covariance_is_plain_squared_distance()
    test_covariance_scales_each_axis()
    test_correlated_covariance()
    test_offset_prediction_and_single_measurement()
    test_gate_admits_the_true_measurement_and_rejects_a_far_one()
    print("all tests passed")
