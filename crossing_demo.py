"""Two crossing targets -- the demo that shows association working.

Run:  python crossing_demo.py

Two targets fly toward each other and pass through the same point. Around the
crossing, both tracks predict nearly the same position, both measurements fall
inside both gates, and the cost matrix is very nearly symmetric. Nothing about
POSITION can tell the tracks apart at that moment.

What separates them is velocity. The filter has spent the whole approach
inferring each target's heading, so the two predictions differ even where the
two positions nearly coincide -- one track expects to keep climbing, the other
to keep descending. That difference is small, but it's enough for the assignment
to come out right, and it's the reason a tracker holds identity through a
crossing when a nearest-blip-wins rule would swap the targets and never recover.

The measurements are shuffled every scan, so the order they arrive in carries no
information about which target produced them. If identity survives, it survives
on the evidence.

ABOUT TRACK CREATION: both tracks are seeded by hand from the first scan, and
which measurement seeds which track is decided here, in the demo, by looking at
ground truth. Deciding that from measurements alone is track initiation, which
tracker.py does and lifecycle_demo.py shows. This demo keeps the hand seeding on
purpose: it isolates association, with no lifecycle in the way.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt

from targets import constant_velocity_track
from sensor import measure
from track import Track
from association import associate, build_cost_matrix
from gating import CHI2_99_2D
from metrics import position_rmse


def main():
    rng = np.random.default_rng(11)

    n_steps = 30
    dt = 1.0
    meas_std = 4.0
    process_var = 0.05

    # two targets on converging paths, crossing at (45, 30) on scan 15
    truth_a = constant_velocity_track(
        x0=0, y0=0, vx=3.0, vy=2.0,
        n_steps=n_steps, dt=dt, accel_std=0.2, rng=rng,
    )[:, [0, 2]]
    truth_b = constant_velocity_track(
        x0=0, y0=60, vx=3.0, vy=-2.0,
        n_steps=n_steps, dt=dt, accel_std=0.2, rng=rng,
    )[:, [0, 2]]

    # each scan reports one measurement per target, in random order
    scans = []
    for k in range(n_steps):
        pair = measure(np.vstack([truth_a[k], truth_b[k]]), meas_std=meas_std,
                       rng=rng)
        scans.append(rng.permutation(pair))

    # seed one track per target from the first scan (see the note above)
    first = scans[0]
    a_first = 0 if (np.linalg.norm(first[0] - truth_a[0])
                    < np.linalg.norm(first[1] - truth_a[0])) else 1
    kwargs = dict(dt=dt, process_var=process_var, meas_var=meas_std**2)
    track_a = Track(first[a_first], track_id=1, **kwargs)
    track_b = Track(first[1 - a_first], track_id=2, **kwargs)
    tracks = [track_a, track_b]

    estimates = {t.id: [t.position] for t in tracks}
    coasted = {t.id: 0 for t in tracks}
    ambiguous_scans = 0

    for k in range(1, n_steps):
        # 1. every track moves to where it expects to be
        for t in tracks:
            t.predict()

        # count the scans where gating settles nothing: every measurement is
        # plausible for every track, so the gate admits both pairings and the
        # assignment is the only thing standing between the tracks and a swap
        if (build_cost_matrix(tracks, scans[k]) <= CHI2_99_2D).all():
            ambiguous_scans += 1

        # 2. decide which measurement goes to which track, all at once
        matches, unmatched_tracks, _ = associate(tracks, scans[k])

        # 3. matched tracks take their measurement; the rest coast
        for track_index, measurement_index in matches:
            tracks[track_index].correct(scans[k][measurement_index])
        for track_index in unmatched_tracks:
            tracks[track_index].correct(None)
            coasted[tracks[track_index].id] += 1

        for t in tracks:
            estimates[t.id].append(t.position)

    estimates = {tid: np.array(v) for tid, v in estimates.items()}

    # --- did each track stay with the target it started on? ---
    truths = {1: truth_a, 2: truth_b}
    # where the paths actually came closest -- the targets wander under their
    # random acceleration, so this is a scan or two off the nominal crossing
    crossing = int(np.argmin(np.linalg.norm(truth_a - truth_b, axis=1)))
    gap = np.linalg.norm(truth_a[crossing] - truth_b[crossing])

    print(f"{n_steps} scans. The targets pass closest at scan {crossing}, "
          f"{gap:.1f} units apart.")
    print("Each track is scored against the target it was seeded on.\n")

    swapped = False
    for t in tracks:
        own = truths[t.id]
        other = truths[2 if t.id == 1 else 1]
        rmse_own = position_rmse(estimates[t.id], own)
        rmse_other = position_rmse(estimates[t.id], other)
        # after the crossing the paths separate again, so the last few scans
        # are where a swap would be unambiguous
        final_gap_own = np.linalg.norm(estimates[t.id][-1] - own[-1])
        final_gap_other = np.linalg.norm(estimates[t.id][-1] - other[-1])
        if final_gap_other < final_gap_own:
            swapped = True

        print(f"track {t.id}:  RMSE vs its own target {rmse_own:5.2f}   "
              f"vs the other {rmse_other:6.2f}   "
              f"coasted {coasted[t.id]} scans")

    print(f"\nOn {ambiguous_scans} of {n_steps - 1} scans every measurement was "
          f"inside every track's gate,\nso gating admitted both pairings and "
          f"the assignment alone decided identity.")

    if swapped:
        print("\nIDENTITY LOST: at least one track finished on the wrong target.")
    else:
        print("\nIdentity held: both tracks finished on the target they started "
              "on, and\ntracking the other target would have cost an order of "
              "magnitude more error.")

    # --- plot ---
    all_measurements = np.vstack(scans)
    plt.figure(figsize=(9, 7))
    plt.plot(truth_a[:, 0], truth_a[:, 1], "g-", lw=2, label="truth: target A")
    plt.plot(truth_b[:, 0], truth_b[:, 1], "g--", lw=2, label="truth: target B")
    plt.scatter(all_measurements[:, 0], all_measurements[:, 1], c="red", s=14,
                alpha=0.4, label="measurements (unlabeled)")
    plt.plot(estimates[1][:, 0], estimates[1][:, 1], "b-", lw=1.8,
             label="track 1")
    plt.plot(estimates[2][:, 0], estimates[2][:, 1], "m-", lw=1.8,
             label="track 2")

    plt.scatter(*truth_a[crossing], s=140, facecolors="none",
                edgecolors="black", lw=1.5, label=f"crossing (scan {crossing})")

    plt.legend()
    plt.title("Two crossing targets, tracked through the crossing\n"
              "measurements arrive unlabeled and in random order")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("crossing_result.png", dpi=120)
    print("\nSaved plot -> crossing_result.png")


if __name__ == "__main__":
    main()
