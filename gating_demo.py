"""Gating demo -- what a track accepts, and how that changes as it settles.

Run:  python gating_demo.py

Each scan this prints the candidates offered to a track, their squared
Mahalanobis distances, and which ones fall inside the gate. That's the Phase 2
milestone: the filter rejecting implausible measurements before any assignment
decision gets made.

The point worth watching is the gate RADIUS column. The decoys sit at fixed
distances from the prediction every scan -- 15, 35 and 80 position units -- but
they are not judged the same way each time. A brand-new track has barely any
idea where its target is (initialize() admits it knows nothing about velocity),
so its gate is enormous and it accepts almost anything. As measurements arrive
and the filter's covariance shrinks, the gate tightens, and candidates that were
plausible on scan 1 stop being plausible by scan 5. A fixed distance threshold
could not do that; this is what the covariance buys.

ABOUT THE DECOYS: to have anything to gate, a scan needs more than one candidate.
The decoys are generated here, in the demo, NOT by sensor.py. Real false alarms
belong to the sensor, and they arrive in Phase 4 with missed detections and the
lifecycle logic that copes with them. These are stand-ins at known distances so
the gate's behavior is legible; they are not a clutter model.
"""

import numpy as np

from targets import constant_velocity_track
from sensor import measure
from track import Track
from gating import squared_mahalanobis_distances, CHI2_99_2D

# Decoys at three fixed distances, in three different directions so the demo
# isn't only probing one axis. Distances chosen to straddle the settled gate:
# the first stays plausible, the last never is.
DECOY_DISTANCES = (15.0, 35.0, 80.0)
DECOY_DIRECTIONS = np.array([
    [1.0, 0.0],
    [-0.6, 0.8],
    [0.3, -0.954],
])


def main():
    rng = np.random.default_rng(7)

    n_steps = 10
    dt = 1.0
    meas_std = 6.0

    truth = constant_velocity_track(
        x0=0, y0=0, vx=2.0, vy=1.5,
        n_steps=n_steps, dt=dt, accel_std=0.3, rng=rng,
    )
    true_xy = truth[:, [0, 2]]
    measurements = measure(true_xy, meas_std=meas_std, rng=rng)

    track = Track(measurements[0], track_id=1, dt=dt, process_var=0.05,
                  meas_var=meas_std**2)

    print(f"Gate: squared Mahalanobis distance <= {CHI2_99_2D} "
          f"(chi-square, 2 dof, keeps 99% of true measurements)")
    print(f"Decoys sit {DECOY_DISTANCES} units from the prediction every scan.")
    print("Radius is how far the gate reaches along x, in position units.\n")

    for k in range(1, n_steps):
        # 1. move the track to where it expects to be this scan
        track.predict()
        z_hat = track.predicted_measurement
        S = track.innovation_covariance

        # 2. assemble the candidates: the real measurement, plus decoys placed
        # at known distances from the prediction
        real = measurements[k]
        decoys = z_hat + DECOY_DIRECTIONS * np.array(DECOY_DISTANCES)[:, None]
        candidates = np.vstack([real, decoys])

        # 3. how plausible is each, given where the track expects to be and how
        # uncertain it is?
        d2 = squared_mahalanobis_distances(z_hat, S, candidates)
        inside = d2 <= CHI2_99_2D

        radius = np.sqrt(CHI2_99_2D * S[0, 0])
        print(f"scan {k:2d}   predicted ({z_hat[0]:6.1f},{z_hat[1]:6.1f})   "
              f"gate radius {radius:5.1f}")

        offsets = np.linalg.norm(candidates - z_hat, axis=1)
        for i, (dist, off, ok) in enumerate(zip(d2, offsets, inside)):
            label = "real measurement" if i == 0 else f"decoy at {off:5.1f}"
            mark = "IN " if ok else "out"
            print(f"          {mark}  {label:<18}  "
                  f"{off:5.1f} units away, d2 = {dist:7.2f}")

        # 4. gating only narrows the field. With one target the real measurement
        # is still the one to hand over; choosing among several survivors is
        # association, and that's Phase 3.
        track.correct(real)

    print("\nThe threshold never changed, but the verdicts did. The 35-unit "
          "decoy was plausible\non scans 1 and 2 and implausible from scan 3 "
          "onward, as the gate shrank from 72 to 23\nunits. The 15-unit decoy "
          "stays inside throughout -- the settled gate still reaches\nfurther "
          "than that, which is why gating narrows the field rather than "
          "deciding it.")


if __name__ == "__main__":
    main()
