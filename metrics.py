"""Scoring the tracker against ground truth.

Whether a tracker "looks like it works" matters less than having a measured
number that can be defended. RMSE (root mean squared error) between estimated
and true position is the standard yardstick: the average distance between where
the tracker thought the target was and where it actually was, in position units.
"""

import numpy as np


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
