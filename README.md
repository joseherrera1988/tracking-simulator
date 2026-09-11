# Multi-Target Tracking Simulator

A modular Python simulation of radar target tracking, state estimation, measurement gating, and multi-target data association.

The system simulates targets moving through space, observes them through a noisy position-only sensor, and reconstructs their trajectories using Kalman filtering. It then extends the single-target tracker to multiple targets using Mahalanobis-distance gating and global nearest-neighbor association solved with the Hungarian algorithm.

The project explores the estimation, tracking, and sensor-fusion problems that arise in real-world radar and autonomous systems.

## Results (Single-Target Tracking)

The baseline experiment runs a target for 60 scans with position measurement noise of σ = 6.0 units per axis.

| Metric | Value |
|---|---|
| Raw measurement RMSE | 8.51 |
| Kalman track RMSE | 4.20 |
| Error reduction vs raw measurements | 50.7% |

The Kalman filter reduces position error by approximately half compared with the raw sensor measurements.

![tracking result](tracking_result.png)

The plot shows:

1. Green: ground-truth target position
2. Red: noisy sensor measurements
3. Blue: Kalman filter estimate

The filter produces a substantially smoother trajectory because it combines noisy measurements with a constant-velocity motion model.

## Run it

```bash
pip install -r requirements.txt
python main.py          # single-target demo, prints RMSE, writes tracking_result.png
python gating_demo.py   # per-scan gate: what a track accepts and rejects
python crossing_demo.py # two crossing targets, writes crossing_result.png
python lifecycle_demo.py # targets come and go under clutter, writes lifecycle_result.png
pytest                  # the full test suite
```

## Design

```
targets.py       ground-truth target motion (constant-velocity model)
sensor.py        noisy radar: measurement noise, missed detections, clutter
kalman.py        constant-velocity Kalman filter (state [x, vx, y, vy])
track.py         one track: a filter plus its ID, status and hit/miss counts
gating.py        Mahalanobis distance from a prediction to candidate measurements
association.py   global nearest neighbor: which measurement belongs to which track
tracker.py       track lifecycle: initiation, confirmation, deletion
metrics.py       RMSE scoring vs ground truth
main.py          single-target end-to-end demo
gating_demo.py   per-scan printout of what the gate accepts and rejects
crossing_demo.py two crossing targets, tracked through the crossing
lifecycle_demo.py targets entering and leaving under clutter, plus a threshold sweep
test_kalman.py   known-answer test on the filter + a noise sanity check
test_track.py    equivalence test: a track matches the bare filter
test_gating.py   hand-computed distances, including a correlated covariance
test_association.py  the case where taking each track's nearest gets it wrong
test_sensor.py   detect() reproduces measure() exactly when nothing is missed
test_tracker.py  each lifecycle rule pinned to the scan it fires on
```

The sensor reports position only; velocity is never measured. The filter infers
it from the position history — visible in the plot as the estimate settling onto
the true heading after the first few scans.

### Key parameters

- `meas_var` — sensor noise variance; should match the real sensor.
- `process_var` — how much target maneuvering the filter expects. Low values
  smooth hard but lag on turns; high values react fast but track noise. Tuning
  this tradeoff is the core of getting good filter behavior.

## Status & roadmap

Phases 1 through 4 of [ROADMAP.md](ROADMAP.md) are done: the filter runs inside a
`Track`, each track can say which measurements are plausible for it, multiple
tracks are assigned their measurements together rather than one at a time, and
tracks are started, confirmed and deleted from the measurements alone under
missed detections and clutter. Multi-target metrics are next.

## Multi-target: two crossing targets

`python crossing_demo.py` runs two targets onto converging paths that pass within
5 units of each other, with the measurements shuffled each scan so their order
carries no clue about which target produced them.

| Metric | Track 1 | Track 2 |
|---|---|---|
| RMSE vs its own target | 3.04 | 2.61 |
| RMSE vs the other target | 29.14 | 28.24 |

Both tracks finish on the target they started on. On 10 of 29 scans every
measurement fell inside every track's gate, so gating admitted both pairings and
the assignment alone kept the identities apart.

![crossing result](crossing_result.png)

What separates the tracks at the crossing is not position, which is nearly
identical, but velocity: each filter has spent the approach inferring its
target's heading, so the predictions differ even where the positions do not.

Association is global nearest neighbor — the pairing with the lowest total cost
across all tracks, solved exactly with the Hungarian algorithm. Taking each
track's nearest measurement in turn is cheaper and gets contested cases wrong;
`test_association.py` pins a case where it costs three times as much. The known
limitation of this approach is that it commits to one hard assignment per scan,
so a confident wrong choice cannot be revisited, where a probabilistic tracker
would carry the ambiguity forward.

## Track lifecycle: targets that come and go

`python lifecycle_demo.py` runs three targets through a 200 × 200 region at
different times. The sensor reports each target with probability 0.9 per scan
and adds two false alarms per scan on average, scattered uniformly. No track is
seeded by hand. Every measurement that no track claims starts a tentative track;
a tentative track is confirmed after a set number of hits and deleted on its
first miss; a confirmed track is deleted after five misses in a row.
Association runs confirmed tracks first, and tentative tracks compete only for
the measurements left over.

In the seeded run, 76 tracks are created. Three are confirmed, one per target,
and each follows its target for as long as the target is present; B's track is
deleted five scans after B leaves. The other 73 are never confirmed, and every
one that ended within the run was deleted within three scans. B also shows the
cost of deleting tentative tracks on their first miss: the sensor missed B on
scans 11 and 14, each miss removed the tentative track B had started, and B was
confirmed only at scan 18, eight scans after it entered.

![lifecycle result](lifecycle_result.png)

One run cannot show a tradeoff, so the demo also sweeps the two thresholds over
20 seeds:

| confirm_after | delete_tentative_after | False tracks per run | Extra tracks on real targets per run | Scans to confirm (mean, max) |
|---|---|---|---|---|
| 3 | 1 | 2.45 | 0.00 | 3.20, 10 |
| 3 | 2 | 6.60 | 0.10 | 3.48, 11 |
| 4 | 1 | 0.15 | 0.00 | 4.70, 11 |
| 4 | 2 | 0.50 | 0.00 | 4.60, 12 |

An extra track on a real target means the target changed track ID during the
run.

The number of hits required to confirm is the setting that controls false
tracks: moving from three to four cuts them by a factor of about sixteen, at a
cost of roughly one and a half scans of delay. Letting a tentative track survive
one miss was expected to reduce restarts on real targets. At this detection
probability it shortens mean confirmation delay by at most 0.1 scans and lengthens
it at three hits, and it roughly triples the false tracks, because clutter-seeded
tracks live long enough to find more clutter.

Giving confirmed tracks first claim on measurements was added after the demo
showed the problem it solves. With all tracks competing equally, a clutter
point pulled target A's track slightly off course at scan 17, a tentative track
started from A's own measurement won the next one, and A changed track ID. Across
the 20 seeds, equal competition left between 0.45 and 0.90 extra tracks per run
on real targets; confirmed-first leaves at most 0.10. It also reduces false
tracks by between a third and two thirds, depending on the thresholds. The
equal-competition figures are from the Phase 4 tracker (commit 2c57d47), scored
with the same sweep and the same seeds.

One limitation is known. A new track's velocity prior, inherited from the
single-target filter, is wide (a standard deviation of about 22 units per scan
against target speeds between 3 and 4), so a new track's gate is large on its
second scan and readily catches clutter.

## Deliberately not built yet

The crossing demo still seeds its two tracks by hand, deliberately, so that it
tests association with no lifecycle logic involved. The velocity prior is left
unchanged because the single-target baseline depends on it, and in the sweep it
mattered less than the confirmation threshold.
