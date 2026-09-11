"""Tests for the scoring functions.

Run with pytest:   pytest test_metrics.py
Or just directly:  python test_metrics.py

GOSPA's value is only as trustworthy as the arithmetic behind it, and a wrong
metric doesn't crash -- it just reports a plausible number. So each test below
is a case small enough to work out by hand, with the working in the docstring.
c = 10 throughout, so a missed target or false track costs c^2 / 2 = 50.
"""

import numpy as np

from metrics import gospa

C = 10.0
NOTHING = np.empty((0, 2))


def _check_decomposition(result):
    """distance^2 must equal localization + c^2/2 * (missed + false)."""
    expected = result["localization"] + C**2 / 2 * (result["missed"]
                                                    + result["false"])
    assert np.isclose(result["distance"] ** 2, expected), result


def test_nothing_against_nothing_is_zero():
    result = gospa(NOTHING, NOTHING, c=C)
    assert result == {"distance": 0.0, "localization": 0.0, "assigned": 0,
                      "missed": 0, "false": 0}


def test_a_perfect_estimate_is_zero():
    result = gospa([[3.0, 4.0]], [[3.0, 4.0]], c=C)
    assert result["distance"] == 0.0
    assert result["assigned"] == 1


def test_localization_error_is_plain_distance():
    """One estimate 3-4-5 away from its truth: d^2 = 25, so d = 5."""
    result = gospa([[3.0, 4.0]], [[0.0, 0.0]], c=C)
    assert np.isclose(result["distance"], 5.0), result
    assert (result["missed"], result["false"]) == (0, 0)
    _check_decomposition(result)


def test_a_missed_target_costs_half_c_squared():
    """No estimates, one truth: d^2 = c^2 / 2 = 50."""
    result = gospa(NOTHING, [[0.0, 0.0]], c=C)
    assert np.isclose(result["distance"], np.sqrt(50.0)), result
    assert (result["missed"], result["false"]) == (1, 0)


def test_a_false_track_costs_half_c_squared():
    """One estimate, no truths: the mirror image, also d^2 = 50."""
    result = gospa([[0.0, 0.0]], NOTHING, c=C)
    assert np.isclose(result["distance"], np.sqrt(50.0)), result
    assert (result["missed"], result["false"]) == (0, 1)


def test_an_estimate_beyond_the_cutoff_is_a_miss_and_a_false_track():
    """An estimate 100 away is not a large localization error -- it is not
    tracking that target at all. It's scored as one miss plus one false track:
    d^2 = 50 + 50 = c^2, so d = c, however far away the estimate is."""
    result = gospa([[100.0, 0.0]], [[0.0, 0.0]], c=C)
    assert np.isclose(result["distance"], C), result
    assert result["assigned"] == 0
    assert (result["missed"], result["false"]) == (1, 1)
    assert result["localization"] == 0.0


def test_the_pairing_is_the_best_total_not_each_nearest():
    """The greedy trap from test_association.py, in metric form. Truths at
    (1, 0) and (-1, 0); estimates at (0, 0) and (3, 0); squared distances

                     truth (1,0)   truth (-1,0)
        est (0,0)         1             1
        est (3,0)         4            16

    Pairing (0,0) with (1,0) first leaves (3,0) with (-1,0): 1 + 16 = 17.
    The best total pairs (3,0) with (1,0) and (0,0) with (-1,0): 4 + 1 = 5.
    A wide cutoff, so this tests the assignment and not the cap."""
    result = gospa([[0.0, 0.0], [3.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]],
                   c=100.0)
    assert np.isclose(result["distance"], np.sqrt(5.0)), result


def test_the_cutoff_changes_which_pairing_is_best():
    """Why the cost matrix is capped at c BEFORE solving, not just filtered
    after. Truths t1 = (0, 0), t2 = (1, 9); estimates e1 = (1, 0),
    e2 = (0, -9.5). Squared distances:

                 t1       t2
        e1        1       81
        e2    90.25   343.25

    Pairing A (e1-t1, e2-t2) costs 1 + 343.25 uncapped, but e2-t2 is beyond
    the cutoff, so it's really a miss plus a false track: 1 + 50 + 50 = 101.
    Pairing B (e1-t2, e2-t1), both inside the cutoff, costs 81 + 90.25 = 171.25.
    GOSPA is the better of the two, A at 101. An uncapped solve would prefer B
    and report 171.25."""
    result = gospa([[1.0, 0.0], [0.0, -9.5]], [[0.0, 0.0], [1.0, 9.0]], c=C)

    assert np.isclose(result["distance"], np.sqrt(101.0)), result
    assert result["assigned"] == 1
    assert (result["missed"], result["false"]) == (1, 1)


def test_a_mixed_scan_decomposes():
    """Two truths, three estimates: one at 3-4-5 from the first truth, one
    exactly on the second, and one false. d^2 = 25 + 0 + 50 = 75."""
    estimates = [[3.0, 4.0], [50.0, 50.0], [-40.0, 0.0]]
    truths = [[0.0, 0.0], [50.0, 50.0]]
    result = gospa(estimates, truths, c=C)

    assert np.isclose(result["distance"], np.sqrt(75.0)), result
    assert result["assigned"] == 2
    assert (result["missed"], result["false"]) == (0, 1)
    assert np.isclose(result["localization"], 25.0)
    _check_decomposition(result)


def test_swapping_the_sets_swaps_missed_and_false():
    """GOSPA is a metric, so it's symmetric; what changes is only which
    leftover is called missed and which false."""
    a = [[3.0, 4.0], [50.0, 50.0], [-40.0, 0.0]]
    b = [[0.0, 0.0], [50.0, 50.0]]
    forward = gospa(a, b, c=C)
    backward = gospa(b, a, c=C)

    assert np.isclose(forward["distance"], backward["distance"])
    assert (forward["missed"], forward["false"]) == (
        backward["false"], backward["missed"])


if __name__ == "__main__":
    # lets you run the file directly without pytest installed
    test_nothing_against_nothing_is_zero()
    test_a_perfect_estimate_is_zero()
    test_localization_error_is_plain_distance()
    test_a_missed_target_costs_half_c_squared()
    test_a_false_track_costs_half_c_squared()
    test_an_estimate_beyond_the_cutoff_is_a_miss_and_a_false_track()
    test_the_pairing_is_the_best_total_not_each_nearest()
    test_the_cutoff_changes_which_pairing_is_best()
    test_a_mixed_scan_decomposes()
    test_swapping_the_sets_swaps_missed_and_false()
    print("all tests passed")
