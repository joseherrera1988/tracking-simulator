# Multi-Target Tracking Simulator

A radar target-tracking pipeline: simulate targets moving through space, observe
them through a noisy sensor, and recover their tracks with Kalman filtering and
(planned) multi-target data association. Built to explore the estimation and
sensor-fusion problems behind real combat-system tracking.

## Results (single-target baseline)

Over a 60-scan run with per-axis measurement noise of σ = 6.0 position units:

| Metric | Value |
|---|---|
| Raw measurement RMSE | 8.51 |
| Kalman track RMSE | **4.20** |
| Error reduction vs raw measurements | **50.7%** |

The filter roughly halves position error versus the raw sensor blips it's given.

![tracking result](tracking_result.png)

The green line is ground truth, red dots are the noisy measurements the tracker
actually receives, and blue is the filter's estimate. The estimate stays smooth
and close to truth despite scattered measurements because the constant-velocity
motion model lets it reject noise inconsistent with plausible target motion.

## Run it

```bash
pip install -r requirements.txt
python main.py        # runs the demo, prints RMSE, writes tracking_result.png
python test_kalman.py # runs the tests (or: pytest test_kalman.py)
```

Phase 3 of the roadmap adds a `scipy` dependency for the assignment solver; it
is not needed yet.

## Design

```
targets.py       ground-truth target motion (constant-velocity model)
sensor.py        noisy radar: measurement noise
kalman.py        constant-velocity Kalman filter (state [x, vx, y, vy])
track.py         one track: a filter plus its ID and hit/miss bookkeeping
metrics.py       RMSE scoring vs ground truth
main.py          single-target end-to-end demo
test_kalman.py   known-answer test on the filter + a noise sanity check
test_track.py    equivalence test: a track matches the bare filter
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

Single-target tracking is complete, tested, and validated. Phase 1 of
[ROADMAP.md](ROADMAP.md) is done: the filter now runs inside a `Track` object
that carries an ID and hit/miss counts, so the pipeline can hold several targets
at once. Gating is next, followed by data association.

Track status and the confirm/delete thresholds are deliberately not implemented
yet. Nothing reads them until Phase 4, which is also where missed detections and
clutter enter the sensor — the conditions those thresholds need to be tuned
against.
