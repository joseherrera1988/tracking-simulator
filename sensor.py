"""A simulated radar.

The real world gives a tracker imperfect information, and modeling that
imperfection honestly is most of what makes this project realistic. For the
single-target starter the imperfection is just NOISE: the reported position is
the true position plus random error each scan. The tracker's whole job is to see
through that noise.

Two other kinds of imperfection matter for multi-target tracking and get added
later, in Phase 4 (see ROADMAP.md), because that's the first point they're
actually needed:
  - missed detections: a real target that isn't reported this scan.
  - false alarms (clutter): a blip reported where no target exists.
Adding them now would be building for a future that doesn't exist yet, so the
sensor stays deliberately simple until the multi-target machinery calls for it.
"""

import numpy as np


def measure(true_positions, meas_std=5.0, rng=None):
    """Add measurement noise to true position(s).

    Args:
        true_positions: either one [x, y] or an (n, 2) array of true positions.
        meas_std: std-dev of the measurement noise, per axis.
        rng: numpy random generator (pass one for reproducible runs).

    Returns:
        (n, 2) array of noisy measurements, same shape as the input positions.
        A single [x, y] input comes back as a (1, 2) array so callers can treat
        the output uniformly.
    """
    if rng is None:
        rng = np.random.default_rng()

    # atleast_2d turns a single [x, y] into a (1, 2) array, so the same code
    # handles one position or many without a special case.
    true_positions = np.atleast_2d(true_positions)
    noise = rng.normal(0, meas_std, size=true_positions.shape)
    return true_positions + noise
