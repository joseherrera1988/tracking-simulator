"""A simulated radar.

The real world gives a tracker imperfect information, and modeling that
imperfection honestly is most of what makes this project realistic. For the
single-target starter the imperfection is just NOISE: the reported position is
the true position plus random error each scan. The tracker's whole job is to see
through that noise.

Two other kinds of imperfection matter for multi-target tracking, and detect()
adds them on top of measure():
  - missed detections: a real target that isn't reported this scan.
  - false alarms (clutter): a blip reported where no target exists.
They arrived in Phase 4 (see ROADMAP.md), with the track lifecycle logic that
has to survive them. measure() is left exactly as it was, for the reason given
in detect().
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


def detect(true_positions, p_detect=1.0, clutter_rate=0.0, region=None,
           meas_std=5.0, rng=None):
    """One radar scan: noisy detections of the targets that were seen, plus
    false alarms, in no particular order.

    Args:
        true_positions: (n, 2) array of true positions of the targets present
                        this scan (may be empty).
        p_detect:     probability that each target is reported this scan.
        clutter_rate: expected number of false alarms per scan (Poisson mean).
        region:       (xmin, xmax, ymin, ymax) the false alarms are scattered
                      over, uniformly. Only read when clutter is drawn.
        meas_std:     std-dev of the measurement noise on real detections.
        rng:          numpy random generator (pass one for reproducible runs).

    Returns:
        (m, 2) array of reported positions, possibly empty. Nothing in it says
        which rows are real -- that's the tracker's problem.

    WHY THIS IS A SEPARATE FUNCTION rather than new arguments on measure(): the
    detection roll and the clutter count are random draws, and a draw advances
    the generator even when it changes nothing -- p_detect=1.0 still consumes
    one. Put inside measure(), every later noise sample would come from a
    different point in the stream, and the seeded runs whose numbers serve as
    regression checks (main.py's 8.51 / 4.20, the crossing demo) would move
    with no bug anywhere. So measure() is untouched, and this function calls it
    FIRST: with the same seed, the noisy positions here are exactly the ones
    measure() would have returned, and only the draws after them are new.
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy = measure(true_positions, meas_std=meas_std, rng=rng)

    # Every target gets noise, even the ones about to be dropped, so that the
    # noise draws line up with measure()'s (see above).
    seen = rng.random(len(noisy)) < p_detect
    detections = noisy[seen]

    n_clutter = rng.poisson(clutter_rate)
    if n_clutter > 0:
        xmin, xmax, ymin, ymax = region
        clutter = rng.uniform((xmin, ymin), (xmax, ymax), size=(n_clutter, 2))
        detections = np.vstack([detections, clutter])

    # Shuffle, or the row order would leak information: real detections first
    # and clutter after is a pattern association could exploit without earning
    # it, and it would make the tracker look better than it is.
    return rng.permutation(detections)
