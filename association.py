"""Which measurement belongs to which track?

Gating (gating.py) narrows the field: for each track it says which measurements
are plausible at all. That is usually not enough. Two tracks flying close
together will both find the same measurement plausible, and only one of them can
have it. Association is the decision that gating deliberately does not make.

The greedy approach -- walk the tracks in order, give each its nearest
unclaimed measurement -- fails exactly when it matters. Whichever track happens
to be considered first takes the contested measurement, and the second is left
with whatever remains, so the outcome depends on track ordering rather than on
the evidence. Two targets crossing is the case that breaks it.

So instead of choosing per track, choose the set of assignments that minimizes
TOTAL distance across all tracks at once. That's an assignment problem, and
scipy.optimize.linear_sum_assignment solves it exactly (the Hungarian algorithm)
in one call. Pairing tracks with measurements by global agreement rather than by
iteration order is what "global nearest neighbor" means.

This is the standard GNN baseline. It commits to one hard assignment per scan,
which is its known weakness: a confident wrong choice is unrecoverable, where a
probabilistic tracker (JPDA, MHT) would carry the ambiguity forward. Sufficient
here, and the thing to say out loud about it.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from gating import squared_mahalanobis_distances, CHI2_99_2D


def build_cost_matrix(tracks, measurements):
    """Squared Mahalanobis distance from every track to every measurement.

    Args:
        tracks: sequence of Tracks, already predicted for this scan.
        measurements: (n, 2) array of this scan's measurements.

    Returns:
        (n_tracks, n_measurements) array. Row i is track i's view of every
        measurement, which is exactly what gating already computes -- the cost
        matrix is that calculation done for each track in turn.
    """
    cost = np.zeros((len(tracks), len(measurements)))
    for i, track in enumerate(tracks):
        cost[i] = squared_mahalanobis_distances(
            track.predicted_measurement, track.innovation_covariance,
            measurements,
        )
    return cost


def associate(tracks, measurements, threshold=CHI2_99_2D):
    """Assign measurements to tracks by global nearest neighbor.

    Args:
        tracks: sequence of Tracks, already predicted for this scan.
        measurements: (n, 2) array of this scan's measurements.
        threshold: gate, as a squared Mahalanobis distance.

    Returns:
        (matches, unmatched_tracks, unmatched_measurements), where matches is a
        list of (track_index, measurement_index) pairs and the other two are
        lists of indices left over. Callers correct() the matched tracks and
        coast the rest.
    """
    cost = build_cost_matrix(tracks, measurements)

    # Forbidden pairs get a large finite cost rather than np.inf: the solver
    # rejects a matrix it cannot fully assign, and with unequal counts it must
    # use some forbidden cell. A sentinel lets it produce an answer, and the
    # gate is enforced afterwards by discarding any pair that exceeds it. The
    # value only needs to be worse than every real option, and every real option
    # is at most `threshold`.
    sentinel = threshold * 1e6
    gated = np.where(cost <= threshold, cost, sentinel)

    track_indices, measurement_indices = linear_sum_assignment(gated)

    matches = [
        (int(t), int(m))
        for t, m in zip(track_indices, measurement_indices)
        if cost[t, m] <= threshold
    ]

    matched_tracks = {t for t, _ in matches}
    matched_measurements = {m for _, m in matches}
    unmatched_tracks = [i for i in range(len(tracks))
                        if i not in matched_tracks]
    unmatched_measurements = [j for j in range(len(measurements))
                              if j not in matched_measurements]

    return matches, unmatched_tracks, unmatched_measurements
