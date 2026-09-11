"""One target track: a Kalman filter plus the bookkeeping around it.

The filter in kalman.py knows how to estimate a state, and nothing else. It has
no identity and no memory of whether the measurements it was fed were plentiful
or sparse. A tracker running several targets at once needs both, because it has
to say WHICH target an estimate belongs to and decide when a track has gone
stale. Track carries that bookkeeping so the filter doesn't have to.

The single-target demo always has a measurement to give, so it never exercises
the coasting path below. Gating and association (Phases 2-3) do: when no
measurement is assigned to a track on a given scan, the track has to keep going
on its prediction alone.

A scan is predict() then correct(), with gating and association in between --
that gap is why the two are separate methods. step() fuses them for the
single-target case, where there is nothing to decide in the middle.

COUNTER SEMANTICS: hits starts at 1, because the measurement that seeds the
filter is itself a hit. The Tracker's confirm/delete thresholds are set against
that baseline -- a track with hits == 3 has been updated twice since being
created. misses counts CONSECUTIVE misses: any hit resets it.

STATUS is "tentative" or "confirmed", and nothing else. There is no "dead": a
deleted track is removed from the Tracker's list, and that removal is the
deletion. A dead status would be a tombstone every caller has to remember to
filter out, and one that could still be handed a measurement by mistake.
"""

from kalman import KalmanFilter2D


class Track:
    def __init__(self, first_measurement, track_id, dt=1.0, process_var=1.0,
                 meas_var=25.0):
        """
        Args:
            first_measurement: [x, y] that creates the track and seeds the filter.
            track_id:          unique identifier, assigned by the caller.
            dt, process_var, meas_var: passed straight through to the filter.

        The ID is supplied rather than generated internally. The Tracker owns
        track creation and hands out IDs; a counter hidden in this class would
        leak state between tests and between Tracker instances.
        """
        self.id = track_id
        self.kf = KalmanFilter2D(dt=dt, process_var=process_var,
                                 meas_var=meas_var)
        self.kf.initialize(first_measurement)

        # The Tracker reads these to decide when a track is confirmed or
        # deleted; the track itself only keeps count.
        self.hits = 1
        self.misses = 0
        self.status = "tentative"

    def predict(self):
        """Move the track forward to this scan's expected position.

        Split out from correct() because gating happens between the two: a scan
        predicts every track first, then judges which measurements are plausible
        for each (which needs the predicted state), and only then decides what to
        assign. A fused predict-and-update leaves nowhere to ask that question.
        """
        return self.kf.predict()

    def correct(self, measurement=None):
        """Fold in whatever this track was assigned this scan.

        Args:
            measurement: [x, y] assigned to this track, or None if nothing was
                         (the track coasts on its prediction).

        Returns:
            The track's position estimate afterwards.
        """
        if measurement is not None:
            self.kf.update(measurement)
            self.hits += 1
            self.misses = 0
        else:
            # No measurement: the prediction stands as the estimate. Note the
            # filter's covariance grew during predict() and nothing shrank it,
            # so a coasting track gets steadily less certain -- which is the
            # honest answer, and why the Tracker deletes one that coasts too long.
            self.misses += 1

        return self.position

    def step(self, measurement=None):
        """Predict and correct in one call.

        Convenience for the single-target case, where there is nothing to gate
        and no assignment to make, so there's no reason to stop in between.
        """
        self.predict()
        return self.correct(measurement)

    @property
    def position(self):
        return self.kf.position

    @property
    def velocity(self):
        return self.kf.velocity

    # Gating reads the two below between predict() and update(), to ask which
    # measurements are plausible for this track. They're delegated for the same
    # reason position and velocity are: callers talk to the Track, and the
    # filter it happens to hold stays an implementation detail.
    @property
    def predicted_measurement(self):
        return self.kf.predicted_measurement

    @property
    def innovation_covariance(self):
        return self.kf.innovation_covariance
