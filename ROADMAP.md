# Roadmap: from single-target to multi-target

The starter code gives you a validated single-target tracker. This is the path
to the full multi-target system. Each phase is a working, demonstrable milestone
on its own — commit at the end of each one so your Git history shows the system
being built up deliberately. **This is the part of the project worth building
yourself; it's what a reviewer reads as your engineering.**

---

## Phase 1 — Track object (refactor) · ~half a day

Right now `main.py` runs one filter inline. Wrap it in a `Track` class so you can
have many at once.

- A `Track` owns: a `KalmanFilter2D`, a unique ID, a hit/miss counter, and a
  status (tentative / confirmed / dead).
- Move the predict/update loop into `Track.step()`.

Milestone: single-target demo works exactly as before, just through the new class.

## Phase 2 — Gating · ~half a day

Before assigning a measurement to a track, decide if it's even *plausible*.

- For each track, predict where it expects a measurement (`H @ x`) and the
  expected spread (innovation covariance `S`, already computed in `update`).
- Compute the Mahalanobis distance from a measurement to that prediction. Accept
  only measurements within a chi-square threshold (for 2D, ~9.21 gates 99%).

Milestone: print, per scan, which measurements fall inside each track's gate.

## Phase 3 — Data association (the core) · ~1–2 days

Multiple tracks, multiple measurements — decide which goes with which. This is
the heart of multi-target tracking.

- Build a cost matrix: rows = tracks, cols = measurements, entry = Mahalanobis
  distance (or ∞ if outside the gate).
- Solve the assignment with `scipy.optimize.linear_sum_assignment` (the Hungarian
  algorithm — one call, don't implement it yourself).
- Update each matched track with its measurement. Leave unmatched tracks to
  coast on prediction only.

This is "global nearest neighbor" (GNN) association. Start here — it's the
standard baseline and entirely sufficient to demo.

Milestone: two crossing targets stay correctly tracked through the crossing.
(The crossing is the money shot — that's the demo that shows association working.)

## Phase 4 — Track lifecycle · ~1 day

Targets appear and disappear; the tracker must too.

- **Initiation**: unmatched measurements that persist across N scans spawn a new
  tentative track; promote to confirmed after M hits.
- **Deletion**: a track that misses measurements for K consecutive scans dies.

This is where you ADD missed detections and clutter to the sensor (a
`p_detect` probability and a `clutter_rate` for false alarms) — they were left
out of the starter on purpose, and this lifecycle logic is exactly what makes
the tracker survive them.

Milestone: targets that enter and leave the scene get tracks created and cleaned
up automatically; clutter doesn't spawn permanent ghost tracks.

## Phase 5 — Metrics & evaluation · ~1 day

Single-target RMSE isn't enough for multi-target. Report:

- Position RMSE over confirmed tracks (extend `metrics.py`).
- Track count accuracy: estimated vs true number of targets over time.
- Optionally read up on the **OSPA** or **GOSPA** metric — the standard
  multi-target tracking score. Citing it shows you know the field.

Milestone: a results table and plots you can drop into the README.

---

## Stretch goals (pick based on time and what you want to signal)

- **Extended Kalman Filter (EKF)**: switch the sensor to report range + bearing
  instead of x/y. Now the measurement model is nonlinear and you must linearize
  it. This is the single most valuable stretch goal — nonlinear estimation is
  core to real radar and shows real depth.
- **C++ port of the filter inner loop** with a benchmark vs Python. Directly
  demonstrates the C/C++ and performance-analysis bullets in the posting.
- **3D tracking** (add z / vz): mostly mechanical once 2D works.
- **CI + tests**: `test_kalman.py` already gives you a known-answer test to
  build on — add more as you go, then wire up GitHub Actions to run them on
  push. Hits the DevSecOps bullet cheaply.
- **IMM (Interacting Multiple Model)**: run constant-velocity and constant-turn
  models in parallel and blend them. Advanced; only if you have lots of time.

## Suggested order for a strong, finite result

Phases 1–5 give you a complete, defensible multi-target tracker. If you have
extra time after that, do the EKF, then add CI + tests. Stop there — a polished
tracker with real metrics beats a sprawling half-finished one, and "I scoped it
and shipped it" is itself the signal you want to send.
