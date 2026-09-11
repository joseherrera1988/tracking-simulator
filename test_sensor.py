"""Tests for the simulated radar.

Run with pytest:   pytest test_sensor.py
Or just directly:  python test_sensor.py

The first test is the one that protects everything else: with detection certain
and no clutter, detect() must report exactly the positions measure() would have,
from the same seed. That's what keeps the seeded demos' numbers meaningful as
regression checks now that the sensor can miss targets and invent them.
"""

import numpy as np

from sensor import measure, detect

TRUE_POSITIONS = np.array([[0.0, 0.0], [50.0, 20.0], [100.0, -40.0]])
REGION = (0.0, 200.0, 0.0, 200.0)


def _sorted_rows(a):
    """Rows in a fixed order, since detect() shuffles its output."""
    return a[np.lexsort(a.T[::-1])]


def test_certain_detection_and_no_clutter_matches_measure():
    """Same seed, same noise: the only difference allowed is the order."""
    reference = measure(TRUE_POSITIONS, meas_std=4.0,
                        rng=np.random.default_rng(5))
    detected = detect(TRUE_POSITIONS, p_detect=1.0, clutter_rate=0.0,
                      meas_std=4.0, rng=np.random.default_rng(5))

    assert detected.shape == reference.shape
    assert np.array_equal(_sorted_rows(detected), _sorted_rows(reference)), (
        "detect() drew its noise differently from measure()"
    )


def test_undetected_targets_are_dropped():
    """p_detect = 0 and no clutter: an empty scan, correctly shaped."""
    detected = detect(TRUE_POSITIONS, p_detect=0.0, clutter_rate=0.0,
                      rng=np.random.default_rng(0))
    assert detected.shape == (0, 2), f"got shape {detected.shape}"


def test_clutter_stays_inside_the_region():
    """With no targets at all, everything reported is clutter and must lie in
    the region it was drawn over. Over many scans the average count should sit
    near the requested rate."""
    rng = np.random.default_rng(1)
    counts = []
    for _ in range(500):
        scan = detect(np.empty((0, 2)), clutter_rate=3.0, region=REGION,
                      rng=rng)
        counts.append(len(scan))
        if len(scan):
            assert (scan[:, 0] >= 0).all() and (scan[:, 0] <= 200).all()
            assert (scan[:, 1] >= 0).all() and (scan[:, 1] <= 200).all()

    assert abs(np.mean(counts) - 3.0) < 0.3, f"mean count {np.mean(counts)}"


def test_row_order_carries_no_information():
    """One target far outside the clutter region, so its row is recognizable.
    If clutter were simply appended, the target would always be row 0."""
    rng = np.random.default_rng(2)
    target = np.array([[1000.0, 1000.0]])
    positions_of_target = set()
    for _ in range(50):
        scan = detect(target, clutter_rate=4.0, region=REGION, rng=rng)
        rows = np.where(scan[:, 0] > 500)[0]
        positions_of_target.add(int(rows[0]))

    assert len(positions_of_target) > 1, (
        "the real detection always came back in the same row"
    )


if __name__ == "__main__":
    # lets you run the file directly without pytest installed
    test_certain_detection_and_no_clutter_matches_measure()
    test_undetected_targets_are_dropped()
    test_clutter_stays_inside_the_region()
    test_row_order_carries_no_information()
    print("all tests passed")
