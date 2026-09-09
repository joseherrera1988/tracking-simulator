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

COUNTER SEMANTICS: hits starts at 1, because the measurement that seeds the
filter is itself a hit. Phase 4 sets its confirm/delete thresholds against that
baseline -- a track with hits == 3 has been updated twice since being created.
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

        The ID is supplied rather than generated internally. In Phase 4 a manager
        owns track creation and is the natural place to hand out IDs; a counter
        hidden in this class would leak state between tests in the meantime.
        """
        self.id = track_id
        self.kf = KalmanFilter2D(dt=dt, process_var=process_var,
                                 meas_var=meas_var)
        self.kf.initialize(first_measurement)

        # Bookkeeping only -- nothing acts on these yet. Phase 4 reads them to
        # decide when a track is confirmed or dead.
        self.hits = 1
        self.misses = 0

    def step(self, measurement=None):
        """Advance the track one scan.

        Args:
            measurement: [x, y] assigned to this track this scan, or None if
                         nothing was assigned (the track coasts).

        Returns:
            The track's position estimate after the step.
        """
        self.kf.predict()

        if measurement is not None:
            self.kf.update(measurement)
            self.hits += 1
            self.misses = 0
        else:
            # No measurement: the prediction stands as the estimate. Note the
            # filter's covariance grew during predict() and nothing shrank it,
            # so a coasting track gets steadily less certain -- which is the
            # honest answer, and what Phase 4 uses to justify deleting it.
            self.misses += 1

        return self.position

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
