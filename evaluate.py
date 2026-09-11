"""How good is the multi-target tracker, and how does it degrade? -- Phase 5.

Run:  python evaluate.py

Every scan is scored with GOSPA (see metrics.gospa): the tracker's confirmed
track positions against the true positions of the targets present. GOSPA
charges for three things at once -- localization error on matched pairs, each
missed target, and each false track -- and reports them separately, so a score
can be taken apart and explained.

THE BASELINE. A GOSPA number on its own means little. The single-target result
is convincing because the raw measurements score 8.51 and the tracker 4.20;
the multi-target equivalent scores the raw detections as if they were the
estimates. That baseline pays for every clutter point as a false target and
every missed detection as a missed target, and has no confirmation delay. The
tracker has to beat it by filtering noise, suppressing clutter and coasting
through misses -- while paying for the scans it spends confirming each target.

WHAT GOSPA CANNOT SEE is identity: it scores each scan on its own, so a tracker
that swapped track IDs every scan could still score well. ID changes are
reported alongside it, from scenario.score(). Trajectory-level GOSPA, which
adds a switching penalty, is the principled fix and is not implemented here.

Two sweeps over 20 seeds each, holding the tracker's thresholds at the
lifecycle demo's settings: clutter rate at fixed p_detect, then p_detect at
fixed clutter rate. Plus a timeline of one seeded run.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt

from metrics import gospa
from scenario import (N_SCANS, P_DETECT, CLUTTER_RATE, simulate,
                      true_positions_at, score)

# Cutoff: above typical localization error (measurement sigma is 4) and below
# the spacing between targets. A missed target or false track costs c^2 / 2.
GOSPA_CUTOFF = 15.0

SEEDS = range(20)
CONFIRM_AFTER = 4                  # the lifecycle demo's settings
DELETE_TENTATIVE_AFTER = 1

CLUTTER_RATES = [0.0, 1.0, 2.0, 3.0, 4.0]   # at p_detect = P_DETECT
P_DETECTS = [1.0, 0.9, 0.8, 0.7]            # at clutter_rate = CLUTTER_RATE

# chart colors: a validated categorical palette, one hue per entity
TRACKER_COLOR = "#2a78d6"
RAW_COLOR = "#eb6834"
PART_COLORS = {"localization": "#1baf7a", "missed": "#eda100",
               "false": "#e87ba4"}
INK = "#52514e"
GRID = "#e1e0d9"


def estimates_at(history, k):
    """(m, 2) array of the confirmed tracks' positions on scan k."""
    positions = [r["positions"][k] for r in history.values()
                 if k in r["positions"]]
    return np.array(positions) if positions else np.empty((0, 2))


def score_scans(truth, scans, history):
    """Per-scan GOSPA results for the tracker and for the raw detections."""
    tracker, raw = [], []
    for k in range(N_SCANS):
        truths = true_positions_at(truth, k)
        tracker.append(gospa(estimates_at(history, k), truths, GOSPA_CUTOFF))
        raw.append(gospa(scans[k], truths, GOSPA_CUTOFF))
    return tracker, raw


def summarize(results):
    """Pool per-scan results into the four numbers reported.

    GOSPA is pooled as the root of the mean squared per-scan distance, so the
    parts stay additive: gospa^2 = localization part + missed part + false part.
    Localization RMSE is over assigned pairs only -- how close the tracks that
    are on a target are to it.
    """
    squared = [r["distance"] ** 2 for r in results]
    assigned = sum(r["assigned"] for r in results)
    localization = sum(r["localization"] for r in results)
    return {
        "gospa": float(np.sqrt(np.mean(squared))),
        "localization_rmse": float(np.sqrt(localization / assigned))
                             if assigned else float("nan"),
        "missed": float(np.mean([r["missed"] for r in results])),
        "false": float(np.mean([r["false"] for r in results])),
    }


def evaluate_setting(p_detect, clutter_rate):
    """All seeds at one sensor setting: tracker and raw summaries, pooled over
    every scan of every seed, mean ID changes per run, and mean scans from a
    target's entry to its first confirmed track."""
    tracker_all, raw_all, id_changes, delays = [], [], [], []
    for seed in SEEDS:
        truth, scans, history = simulate(seed, CONFIRM_AFTER,
                                         DELETE_TENTATIVE_AFTER,
                                         p_detect=p_detect,
                                         clutter_rate=clutter_rate)
        tracker, raw = score_scans(truth, scans, history)
        tracker_all += tracker
        raw_all += raw
        _, extra_tracks, run_delays, _ = score(truth, history)
        id_changes.append(extra_tracks)
        delays += run_delays
    return (summarize(tracker_all), summarize(raw_all),
            float(np.mean(id_changes)), float(np.mean(delays)))


def run_sweep(label, settings):
    """Evaluate each (p_detect, clutter_rate) setting and print a table row."""
    print(f"\n{label}  ({len(SEEDS)} seeds, GOSPA cutoff {GOSPA_CUTOFF:.0f})\n")
    print("  p_detect  clutter |  GOSPA          | localization RMSE | "
          "missed/scan     | false/scan      | ID changes | scans to")
    print("                    |  tracker   raw  |  tracker   raw    | "
          "tracker   raw   | tracker   raw   |  per run   | confirm")
    rows = []
    for p_detect, clutter_rate in settings:
        tracker, raw, id_changes, delay = evaluate_setting(p_detect,
                                                           clutter_rate)
        rows.append((tracker, raw))
        print(f"  {p_detect:8.1f}  {clutter_rate:7.1f} | "
              f"{tracker['gospa']:7.2f} {raw['gospa']:6.2f}  | "
              f"{tracker['localization_rmse']:7.2f} {raw['localization_rmse']:6.2f}"
              f"    | {tracker['missed']:6.2f} {raw['missed']:6.2f}   | "
              f"{tracker['false']:6.2f} {raw['false']:6.2f}   | "
              f"{id_changes:6.2f}     | {delay:6.2f}")
    return rows


def _style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK)


def plot_sweeps(clutter_rows, p_detect_rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    panels = [
        (axes[0], CLUTTER_RATES, clutter_rows,
         f"False alarms per scan (p_detect {P_DETECT})"),
        (axes[1], P_DETECTS, p_detect_rows,
         f"Detection probability ({CLUTTER_RATE:.0f} false alarms per scan)"),
    ]
    for ax, xs, rows, xlabel in panels:
        for i, (name, color) in enumerate([("tracker", TRACKER_COLOR),
                                           ("raw detections", RAW_COLOR)]):
            ys = [row[i]["gospa"] for row in rows]
            ax.plot(xs, ys, "-o", color=color, lw=2, ms=8,
                    markeredgecolor="white", markeredgewidth=1.5, label=name)
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK,
                        fontsize=9)
        ax.set_xlabel(xlabel, color=INK)
        _style(ax)
    axes[1].invert_xaxis()   # worse detection to the right, like more clutter
    axes[0].set_ylabel("GOSPA per scan (RMS over scans and seeds)", color=INK)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Multi-target error as the sensor gets worse: tracker vs raw "
                 "detections", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig("evaluation_sweeps.png", dpi=120)


def plot_timeline(seed=7):
    truth, scans, history = simulate(seed, CONFIRM_AFTER,
                                     DELETE_TENTATIVE_AFTER)
    tracker, raw = score_scans(truth, scans, history)
    ks = np.arange(N_SCANS)
    half_c2 = GOSPA_CUTOFF**2 / 2

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1.6]})

    true_count = [len(true_positions_at(truth, k)) for k in ks]
    track_count = [len(estimates_at(history, k)) for k in ks]
    top.step(ks, true_count, where="mid", color=INK, lw=2, label="true targets")
    top.step(ks, track_count, where="mid", color=TRACKER_COLOR, lw=2,
             label="confirmed tracks")
    top.set_ylabel("count", color=INK)
    top.set_yticks(range(0, max(true_count + track_count) + 2))
    top.legend(frameon=False, loc="upper left", ncol=2)
    _style(top)

    parts = {
        "localization": [r["localization"] for r in tracker],
        "missed": [half_c2 * r["missed"] for r in tracker],
        "false": [half_c2 * r["false"] for r in tracker],
    }
    bottom.stackplot(ks, *parts.values(), step="mid",
                     colors=list(PART_COLORS.values()),
                     labels=[f"tracker: {name}" for name in parts],
                     edgecolor="white", linewidth=0.8)
    bottom.step(ks, [r["distance"] ** 2 for r in raw], where="mid",
                color=RAW_COLOR, lw=2, label="raw detections: total")
    bottom.set_ylabel("GOSPA$^2$ per scan (units$^2$)", color=INK)
    bottom.set_xlabel("scan", color=INK)
    bottom.set_ylim(bottom=0)
    bottom.legend(frameon=False, loc="upper right", fontsize=9)
    _style(bottom)

    fig.suptitle(f"One run (seed {seed}): what the tracker's error is made of, "
                 f"scan by scan", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig("evaluation_timeline.png", dpi=120)


def main():
    clutter_rows = run_sweep(
        "Sweep 1: clutter rate",
        [(P_DETECT, rate) for rate in CLUTTER_RATES])
    p_detect_rows = run_sweep(
        "Sweep 2: detection probability",
        [(p, CLUTTER_RATE) for p in P_DETECTS])
    plot_sweeps(clutter_rows, p_detect_rows)
    plot_timeline()
    print("\nSaved plots -> evaluation_sweeps.png, evaluation_timeline.png")


if __name__ == "__main__":
    main()
