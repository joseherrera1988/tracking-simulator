"""Single-target tracking demo -- the working foundation.

Run:  python main.py

This wires the pieces together for ONE target:
  1. generate a true trajectory        (targets.py)
  2. produce noisy measurements         (sensor.py)
  3. run a track over them              (track.py, wrapping kalman.py)
  4. score it and plot the result       (metrics.py)

Once this makes sense to you end to end, ROADMAP.md is your path to the
multi-target version -- which is the part worth putting your name on.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt

from targets import constant_velocity_track
from sensor import measure
from track import Track
from metrics import position_rmse, raw_measurement_rmse


def main():
    rng = np.random.default_rng(42)  # fixed seed = reproducible run

    # --- simulation parameters (tweak these and see what happens) ---
    n_steps = 60
    dt = 1.0
    meas_std = 6.0          # sensor noise
    process_var = 0.05      # how much target maneuvering the filter expects

    # 1. true trajectory: starts at (0,0), moving up-right, gentle wander
    truth = constant_velocity_track(
        x0=0, y0=0, vx=2.0, vy=1.5,
        n_steps=n_steps, dt=dt, accel_std=0.3, rng=rng,
    )
    true_xy = truth[:, [0, 2]]  # pull out the [x, y] columns

    # 2. noisy measurements -- one per scan. The sensor adds noise to every true
    # position at once, so no per-step loop is needed here.
    measurements = measure(true_xy, meas_std=meas_std, rng=rng)

    # 3. run the track, one scan at a time. The first measurement creates the
    # track; every later one is a step. With a single target there's always a
    # measurement to hand over, so this track never coasts.
    track = Track(measurements[0], track_id=1, dt=dt,
                  process_var=process_var, meas_var=meas_std**2)

    estimates = np.zeros((n_steps, 2))
    estimates[0] = track.position
    for k in range(1, n_steps):
        estimates[k] = track.step(measurements[k])

    # 4. score it
    track_rmse = position_rmse(estimates, true_xy)
    meas_rmse = raw_measurement_rmse(measurements, true_xy)

    print(f"Raw measurement RMSE : {meas_rmse:6.2f}")
    print(f"Kalman track RMSE    : {track_rmse:6.2f}")
    print(f"Improvement          : {(1 - track_rmse / meas_rmse) * 100:5.1f}% "
          f"lower error than raw measurements")

    # plot
    plt.figure(figsize=(9, 7))
    plt.plot(true_xy[:, 0], true_xy[:, 1], "g-", lw=2, label="ground truth")
    plt.scatter(measurements[:, 0], measurements[:, 1], c="red", s=18,
                alpha=0.5, label="noisy measurements")
    plt.plot(estimates[:, 0], estimates[:, 1], "b-", lw=1.8,
             label="Kalman estimate")
    plt.legend()
    plt.title(f"Single-target tracking\n"
              f"track RMSE {track_rmse:.2f} vs raw {meas_rmse:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("tracking_result.png", dpi=120)
    print("\nSaved plot -> tracking_result.png")


if __name__ == "__main__":
    main()
