"""A constant-velocity Kalman filter for 2D position tracking.

THE ONE-PARAGRAPH INTUITION
---------------------------
The filter carries a belief about where the target is and how fast it's moving,
plus how uncertain it is about that belief. Each scan it does two things:

  PREDICT: "given my last estimate, where should the target be now?" -- this
           moves the estimate forward and GROWS the uncertainty (time passed,
           the target could have wandered).
  UPDATE:  "a measurement just arrived; blend it with my prediction" -- this
           pulls the estimate toward the measurement and SHRINKS the
           uncertainty. How hard it pulls depends on the Kalman gain, which is
           just the filter deciding who to trust: a noisy sensor (big R) gets
           trusted less, a shaky prediction (big P) gets trusted less.

That's the whole thing. The five equations below are just that idea written in
matrix form so it works in any number of dimensions at once.

STATE LAYOUT: we track [x, vx, y, vy].
MEASUREMENT:  the sensor reports [x, y] only -- never velocity. The filter
              INFERS velocity from how position changes over time. Seeing that
              actually work is the satisfying part.
"""

import numpy as np


class KalmanFilter2D:
    def __init__(self, dt=1.0, process_var=1.0, meas_var=25.0):
        """
        Args:
            dt:          time between scans (seconds).
            process_var: how much random acceleration we EXPECT the target to
                         have. Bigger = filter trusts measurements more and
                         reacts fast but jitters. This is a tuning knob.
            meas_var:    measurement noise variance = (sensor meas_std)**2.
                         Should roughly match the real sensor. Also a tuning knob.

        The interplay between process_var and meas_var is the entire "feel" of
        the filter. Expect to spend real time here -- that's normal, not a bug.
        """
        self.dt = dt

        # F: state transition. Applies constant-velocity physics for one step:
        #    new_x = x + vx*dt ;  vx unchanged  (same for y)
        self.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ], dtype=float)

        # H: measurement matrix. Pulls [x, y] out of the [x, vx, y, vy] state.
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ], dtype=float)

        # R: measurement noise covariance. How noisy the sensor is.
        self.R = np.eye(2) * meas_var

        # Q: process noise covariance. This is the "discrete white noise
        # acceleration" model -- the standard way to express "the target might
        # accelerate randomly with variance process_var" as a 4x4 matrix.
        q = process_var
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        block = np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])
        self.Q = np.zeros((4, 4))
        self.Q[0:2, 0:2] = block * q   # x, vx
        self.Q[2:4, 2:4] = block * q   # y, vy

        # x: current state estimate (unknown until first measurement).
        self.x = np.zeros(4)
        # P: current estimate covariance = how unsure we are. Start large:
        # we know almost nothing yet, so let the first measurements dominate.
        self.P = np.eye(4) * 1000.0

    def initialize(self, first_measurement):
        """Seed the state from the first measurement. Position = measured;
        velocity = unknown (0) but with large uncertainty so it corrects fast."""
        mx, my = first_measurement
        self.x = np.array([mx, 0.0, my, 0.0])
        self.P = np.diag([self.R[0, 0], 500.0, self.R[1, 1], 500.0])

    def predict(self):
        """Move the estimate forward in time and grow the uncertainty."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """Fold in one measurement z = [x, y] and shrink the uncertainty."""
        z = np.asarray(z, dtype=float)
        y = z - self.H @ self.x            # innovation: measurement minus prediction
        S = self.H @ self.P @ self.H.T + self.R   # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain (who to trust)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x

    @property
    def position(self):
        return np.array([self.x[0], self.x[2]])

    @property
    def velocity(self):
        return np.array([self.x[1], self.x[3]])
