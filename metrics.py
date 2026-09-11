"""Scoring the tracker against ground truth.

Whether a tracker "looks like it works" matters less than having a measured
number that can be defended. RMSE (root mean squared error) between estimated
and true position is the standard yardstick: the average distance between where
the tracker thought the target was and where it actually was, in position units.

RMSE needs to know which estimate goes with which truth, and with one target
that's trivial. With several it isn't, and RMSE has no way to charge for the two
other ways a multi-target tracker goes wrong: a target with no track (missed)
and a track with no target (false). gospa() below scores all three at once.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def position_rmse(estimates, truth):
    """RMSE between estimated and true positions.

    Args:
        estimates: (n, 2) array of estimated [x, y].
        truth:     (n, 2) array of true [x, y].

    Returns:
        scalar RMSE in position units.
    """
    estimates = np.asarray(estimates)
    truth = np.asarray(truth)
    sq_err = np.sum((estimates - truth) ** 2, axis=1)  # squared distance per step
    return float(np.sqrt(np.mean(sq_err)))


def raw_measurement_rmse(measurements, truth):
    """RMSE of the raw sensor measurements vs truth, for comparison.

    If your tracker's RMSE isn't clearly BELOW this, the filter isn't earning
    its keep -- you'd do as well just plotting the raw blips. This one line is
    the most convincing sentence in your README.
    """
    return position_rmse(measurements, truth)


def gospa(estimates, truths, c=15.0):
    """GOSPA distance between a set of estimates and a set of true positions,
    for one scan.

    Generalized optimal sub-pattern assignment (Rahmathullah, Garcia-Fernandez
    and Svensson, 2017), with order p = 2 and alpha = 2:

        d^2 = min over assignments of
              [ sum of squared distances of the assigned pairs
                + c^2 / 2 * (number of missed targets + number of false tracks) ]

    Each estimate is paired with at most one truth, and a pair is only allowed
    if it is closer than the cutoff c. Anything left unpaired is a missed
    target (a truth) or a false track (an estimate), and costs c^2 / 2. So a
    pair at distance c or more is never worth making: its cost would be c^2,
    exactly what leaving both unpaired costs.

    Alpha = 2 is what makes the result split cleanly into those three parts,
    which is the reason to prefer GOSPA over OSPA here: OSPA averages over the
    larger set's size and gives one number that can't be taken apart.

    Args:
        estimates: (m, 2) array of estimated positions (m may be 0).
        truths:    (n, 2) array of true positions (n may be 0).
        c:         cutoff distance, in position units. Should be above typical
                   localization error and below the spacing between targets.

    Returns:
        dict with
          "distance":     the GOSPA distance d, in position units
          "localization": sum of squared distances over the assigned pairs
          "assigned":     number of assigned pairs
          "missed":       truths left unassigned
          "false":        estimates left unassigned
        so that distance**2 == localization + c**2 / 2 * (missed + false).
    """
    estimates = np.asarray(estimates, dtype=float).reshape(-1, 2)
    truths = np.asarray(truths, dtype=float).reshape(-1, 2)

    # Every estimate against every truth, capped at the cutoff. The cap means
    # the solver sees a too-distant pair as costing c^2 -- the same as leaving
    # both unpaired -- so it never gains by forcing one.
    diff = estimates[:, None, :] - truths[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))           # (m, n)
    cost = np.minimum(distances, c) ** 2

    rows, cols = linear_sum_assignment(cost)
    kept = distances[rows, cols] < c                       # pairs worth making

    assigned = int(np.sum(kept))
    localization = float(np.sum(distances[rows, cols][kept] ** 2))
    missed = len(truths) - assigned
    false = len(estimates) - assigned

    distance = np.sqrt(localization + c**2 / 2 * (missed + false))
    return {"distance": float(distance), "localization": localization,
            "assigned": assigned, "missed": missed, "false": false}
