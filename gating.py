"""Which measurements are plausible for a track at all?

Before deciding WHICH measurement belongs to which track (that's association,
Phase 3), it's worth throwing out the ones that couldn't belong to a given track
under any reasonable story. That's gating, and it does two jobs: it keeps
obviously-wrong measurements from ever being considered, and it shrinks the
assignment problem association has to solve.

The naive way to ask "is this measurement close to that track?" is straight-line
distance, and it's wrong, because "close" isn't a fixed number of units. A
measurement 10 units from the prediction is unremarkable when the filter is
uncertain and the sensor is noisy; it's damning when both are tight. The
Mahalanobis distance handles that by measuring distance in units of the expected
spread S: it asks how many standard deviations off the measurement is, in a
direction-aware way, rather than how many position units.

    d^2 = (z - z_hat)' S^-1 (z - z_hat)

where z_hat is what the filter predicts it should see and S is the innovation
covariance -- both supplied by the filter (see KalmanFilter2D).

WHAT THIS MODULE DOES NOT DO: pick the threshold. The estimator produces the
primitives, and deciding what counts as "close enough" is a tracking decision
that depends on how much clutter you expect and how much you fear losing a real
target. So the threshold is applied by the caller. CHI2_99_2D below is a
documented default, not a policy baked into the math.
"""

import numpy as np

# Under the assumption that the filter is consistent (its S honestly describes
# its error), d^2 for a measurement that really came from the target follows a
# chi-square distribution with one degree of freedom per measured dimension --
# 2 here, since the sensor reports x and y. So a threshold read off the
# chi-square table gates a chosen fraction of true measurements:
#     9.21 -> keeps 99% of true measurements
#     5.99 -> keeps 95% (tighter: fewer false candidates, more missed real ones)
CHI2_99_2D = 9.21


def squared_mahalanobis_distances(predicted_measurement, innovation_covariance,
                                  measurements):
    """Squared Mahalanobis distance from a prediction to each measurement.

    Args:
        predicted_measurement: [x, y] the filter expects to see this scan.
        innovation_covariance: the 2x2 expected spread S around it.
        measurements: one [x, y] or an (n, 2) array of candidates.

    Returns:
        (n,) array of squared distances, one per measurement. Compare these
        directly against a chi-square threshold such as CHI2_99_2D.

    The name says "squared" because the chi-square thresholds are stated in
    those terms; taking a square root here would invite comparing a distance
    against 9.21 and silently gating far too loosely.

    Takes the two filter quantities as plain arrays rather than a filter or a
    track, so this stays pure math with nothing to reach through. It also means
    the function needs no changes when an EKF supplies a nonlinear prediction
    and a Jacobian-based S -- the arithmetic here is the same either way.
    """
    # atleast_2d so one measurement and many are handled the same way, matching
    # the convention in sensor.measure().
    measurements = np.atleast_2d(measurements)
    y = measurements - predicted_measurement          # (n, 2) innovations

    # Solve S @ u = y' rather than forming S^-1 explicitly. Same answer, but it
    # stays well behaved when S is close to singular -- which happens when the
    # filter becomes very confident along some direction.
    solved = np.linalg.solve(innovation_covariance, y.T)   # (2, n)

    # Row-wise y @ S^-1 @ y: multiply elementwise and sum across each row.
    return np.sum(y * solved.T, axis=1)
