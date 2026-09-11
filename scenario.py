"""The multi-target scenario, and how a run of it is scored against truth.

Three targets cross a 200 x 200 surveillance region at different times: A is
there throughout, B enters late and leaves early, C enters halfway through. A
run simulates the targets, passes them through the sensor, feeds every scan to
a Tracker, and records what happened to each track.

It lives here rather than in lifecycle_demo.py because two scripts run it: the
demo, which shows one run in detail, and evaluate.py, which runs it across
seeds and sensor settings. Ground truth is used ONLY to score a run afterwards;
the tracker never sees it.
"""

import numpy as np

from targets import constant_velocity_track
from sensor import detect
from tracker import Tracker

N_SCANS = 60
REGION = (0, 200, 0, 200)          # xmin, xmax, ymin, ymax
P_DETECT = 0.9
CLUTTER_RATE = 2.0                 # false alarms per scan, on average
MEAS_STD = 4.0
PROCESS_VAR = 0.05

# name, first scan, last scan + 1, start position, velocity
TARGETS = [
    ("A", 0, 60, (10, 20), (3.0, 2.0)),
    ("B", 10, 45, (20, 180), (3.5, -2.0)),
    ("C", 25, 60, (190, 100), (-3.0, 0.5)),
]

# a confirmed track counts as following a target on a scan if it is within
# this distance of it -- about 3.5 sigma of the measurement noise
FOLLOW_DISTANCE = 15.0


def true_positions_at(truth, k):
    """(n, 2) array of the positions of the targets present on scan k."""
    present = [path[k - first] for first, end, path in truth.values()
               if first <= k < end]
    return np.array(present) if present else np.empty((0, 2))


def simulate(seed, confirm_after, delete_tentative_after, p_detect=P_DETECT,
             clutter_rate=CLUTTER_RATE):
    """One run of the scenario.

    Returns:
        truth:   {name: (first_scan, end_scan, (n, 2) true positions)}
        scans:   list of each scan's measurements
        history: {track_id: dict of the scans it was created / confirmed /
                  deleted on, and its positions while confirmed}
    """
    rng = np.random.default_rng(seed)

    truth = {}
    for name, first, end, (x0, y0), (vx, vy) in TARGETS:
        path = constant_velocity_track(x0, y0, vx, vy, n_steps=end - first,
                                       accel_std=0.2, rng=rng)
        truth[name] = (first, end, path[:, [0, 2]])

    tracker = Tracker(confirm_after=confirm_after,
                      delete_tentative_after=delete_tentative_after,
                      delete_confirmed_after=5,
                      process_var=PROCESS_VAR, meas_var=MEAS_STD**2)

    scans = []
    history = {}
    for k in range(N_SCANS):
        measurements = detect(true_positions_at(truth, k), p_detect,
                              clutter_rate, REGION, MEAS_STD, rng)
        scans.append(measurements)

        for event, track_id in tracker.step(measurements):
            record = history.setdefault(track_id, {"positions": {}})
            record[event] = k

        for track in tracker.confirmed_tracks:
            history[track.id]["positions"][k] = track.position

    return truth, scans, history


def followed_target(record, truth):
    """Which target a confirmed track spent most of its confirmed life near,
    or None if it was near no target for most of it (a false track)."""
    votes = []
    for k, position in record["positions"].items():
        for name, (first, end, path) in truth.items():
            if (first <= k < end and np.linalg.norm(position - path[k - first])
                    < FOLLOW_DISTANCE):
                votes.append(name)
                break
    if not votes:
        return None
    name = max(set(votes), key=votes.count)
    return name if votes.count(name) > len(record["positions"]) / 2 else None


def score(truth, history):
    """False confirmed tracks, extra confirmed tracks on real targets (a target
    followed by two tracks over the run changed ID once), and scans from each
    target's entry until a track following it is confirmed."""
    confirmed = {tid: r for tid, r in history.items() if "confirmed" in r}
    labels = {tid: followed_target(r, truth) for tid, r in confirmed.items()}
    false_tracks = sum(1 for name in labels.values() if name is None)
    followed = [name for name in labels.values() if name is not None]
    extra_tracks = len(followed) - len(set(followed))

    delays = []
    for name, (first, _, _) in truth.items():
        confirm_scans = [confirmed[tid]["confirmed"]
                         for tid, n in labels.items() if n == name]
        if confirm_scans:
            delays.append(min(confirm_scans) - first)
    return false_tracks, extra_tracks, delays, labels
