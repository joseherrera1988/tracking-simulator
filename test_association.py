"""Tests for data association.

Run with pytest:   pytest test_association.py
Or just directly:  python test_association.py

The first test is the one that justifies the whole module: a case where taking
each track's nearest measurement in turn gets the answer wrong, and solving for
the best total gets it right. If that test passes, the Hungarian call is earning
its keep; if a greedy loop would do just as well, it isn't.
"""

import numpy as np

from association import build_cost_matrix, associate
from gating import CHI2_99_2D


class _StubTrack:
    """A stand-in exposing only what association reads.

    Real Tracks work here too -- test_association_is_independent_of_order uses
    them -- but their covariance depends on how many measurements they've seen,
    which makes exact costs awkward to state. With S = I the cost is plain
    squared distance, so the arithmetic in these tests is checkable by eye.
    """

    def __init__(self, predicted_measurement):
        self.predicted_measurement = np.asarray(predicted_measurement,
                                                dtype=float)
        self.innovation_covariance = np.eye(2)


def test_global_solution_beats_taking_each_nearest():
    """The case greedy gets wrong.

    Tracks at (1, 0) and (-1, 0); measurements at (0, 0) and (3, 0). With S = I
    the costs are:

              m0    m3
        A  [   1     4 ]
        B  [   1    16 ]

    Handling A first, it takes m0 (cost 1, its nearest), which strands B with
    m3 for 16 -- total 17. Assigning by best total instead gives A the farther
    m3 and B the near m0: total 5. Same evidence, a third of the cost, and the
    difference is entirely down to not deciding one track at a time.
    """
    tracks = [_StubTrack([1.0, 0.0]), _StubTrack([-1.0, 0.0])]
    measurements = np.array([[0.0, 0.0], [3.0, 0.0]])

    cost = build_cost_matrix(tracks, measurements)
    assert np.allclose(cost, [[1.0, 4.0], [1.0, 16.0]]), f"got {cost}"

    # a wide gate, so this tests the assignment and not the threshold
    matches, unmatched_tracks, unmatched_measurements = associate(
        tracks, measurements, threshold=1000.0
    )

    assert sorted(matches) == [(0, 1), (1, 0)], (
        f"expected the global pairing, got {matches}"
    )
    assert unmatched_tracks == []
    assert unmatched_measurements == []


def test_measurements_outside_every_gate_go_unmatched():
    """The gate still applies after the solver runs: a pairing the assignment
    would happily make is thrown out if it isn't plausible."""
    tracks = [_StubTrack([0.0, 0.0])]
    measurements = np.array([[0.5, 0.0], [400.0, 400.0]])

    matches, unmatched_tracks, unmatched_measurements = associate(
        tracks, measurements, threshold=CHI2_99_2D
    )

    assert matches == [(0, 0)], f"got {matches}"
    assert unmatched_tracks == []
    assert unmatched_measurements == [1]


def test_track_with_nothing_plausible_is_left_to_coast():
    """A track whose gate is empty gets no measurement -- the caller coasts it
    rather than forcing it onto whatever was least bad."""
    tracks = [_StubTrack([0.0, 0.0]), _StubTrack([500.0, 500.0])]
    measurements = np.array([[0.4, 0.3]])

    matches, unmatched_tracks, unmatched_measurements = associate(
        tracks, measurements, threshold=CHI2_99_2D
    )

    assert matches == [(0, 0)], f"got {matches}"
    assert unmatched_tracks == [1]
    assert unmatched_measurements == []


def test_no_measurements_at_all():
    """An empty scan leaves every track unmatched rather than raising."""
    tracks = [_StubTrack([0.0, 0.0]), _StubTrack([5.0, 5.0])]
    measurements = np.empty((0, 2))

    matches, unmatched_tracks, unmatched_measurements = associate(
        tracks, measurements
    )

    assert matches == []
    assert unmatched_tracks == [0, 1]
    assert unmatched_measurements == []


def test_association_is_independent_of_order():
    """Real tracks, and the measurements shuffled: the pairing must follow the
    evidence, not the order the sensor happened to report things in."""
    from track import Track

    kwargs = dict(dt=1.0, process_var=0.05, meas_var=16.0)
    a = Track([0.0, 0.0], track_id=1, **kwargs)
    b = Track([40.0, 0.0], track_id=2, **kwargs)
    for _ in range(5):                      # let both covariances settle
        a.step([0.0, 0.0])
        b.step([40.0, 0.0])

    a.predict()
    b.predict()

    near_a = np.array([1.0, 0.5])
    near_b = np.array([39.0, -0.5])

    in_order, _, _ = associate([a, b], np.array([near_a, near_b]))
    reversed_order, _, _ = associate([a, b], np.array([near_b, near_a]))

    assert sorted(in_order) == [(0, 0), (1, 1)], f"got {in_order}"
    assert sorted(reversed_order) == [(0, 1), (1, 0)], f"got {reversed_order}"


if __name__ == "__main__":
    # lets you run the file directly without pytest installed
    test_global_solution_beats_taking_each_nearest()
    test_measurements_outside_every_gate_go_unmatched()
    test_track_with_nothing_plausible_is_left_to_coast()
    test_no_measurements_at_all()
    test_association_is_independent_of_order()
    print("all tests passed")
