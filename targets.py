"""Ground-truth target motion.

A target here is just a physical object moving through 2D space. We generate
its TRUE positions over time. The tracker never sees these directly -- it only
sees noisy measurements (see sensor.py). We keep ground truth around so we can
measure how well the tracker did (see metrics.py).

We use a constant-velocity (CV) motion model: the target moves in a straight
line at fixed speed, with a little random acceleration ("process noise") each
step so it isn't perfectly straight. This is the same model the Kalman filter
assumes, which is a deliberate starting point -- get this working first, then
later make the target maneuver in ways the filter does NOT expect and watch how
the tracker copes. That mismatch is where the interesting engineering lives.
"""

import numpy as np


def constant_velocity_track(
    x0, y0, vx, vy, n_steps, dt=1.0, accel_std=0.0, rng=None
):
    """Generate one target's true trajectory.

    Args:
        x0, y0:   starting position
        vx, vy:   starting velocity (units per second)
        n_steps:  how many time steps to simulate
        dt:       seconds between steps (the "scan interval" of the radar)
        accel_std: std-dev of random acceleration each step. 0 = perfectly
                   straight line. Small values = gentle wander.
        rng:      numpy random generator (pass one for reproducible runs)

    Returns:
        (n_steps, 4) array where each row is [x, vx, y, vy] at that time.
    """
    if rng is None:
        rng = np.random.default_rng()

    state = np.array([x0, vx, y0, vy], dtype=float)
    history = np.zeros((n_steps, 4))

    for k in range(n_steps):
        history[k] = state
        # random acceleration this step (0 if accel_std == 0)
        ax = rng.normal(0, accel_std)
        ay = rng.normal(0, accel_std)
        # advance position by velocity, velocity by acceleration
        state = np.array([
            state[0] + state[1] * dt + 0.5 * ax * dt**2,
            state[1] + ax * dt,
            state[2] + state[3] * dt + 0.5 * ay * dt**2,
            state[3] + ay * dt,
        ])

    return history
