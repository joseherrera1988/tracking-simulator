"""Targets that come and go, a sensor that misses and lies -- the Phase 4 demo.

Run:  python lifecycle_demo.py

Three targets cross a 200 x 200 surveillance region at different times: A is
there throughout, B enters late and leaves early, C enters halfway through. The
sensor reports each target with probability 0.9 per scan, and adds on average
two false alarms per scan scattered across the whole region. Nothing is seeded
by hand: every track is started, confirmed and deleted by the Tracker from the
measurements alone.

Two outputs, answering two different questions:

  1. An EVENT LOG for one run. A plot can show where a track went, but not the
     moment it was confirmed or deleted -- the log is what shows the lifecycle
     actually working. Every clutter point spawns a tentative track, so the log
     summarizes those instead of printing a hundred create/delete pairs.

  2. A SWEEP over 20 seeds of the two thresholds that matter, showing what each
     buys and costs. One seeded run can't show a tradeoff; averages can.

Ground truth is used ONLY to score the result afterwards -- to say which target
a confirmed track was following, or that it was following nothing. The
scenario itself, and that scoring, live in scenario.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt

from scenario import N_SCANS, REGION, P_DETECT, CLUTTER_RATE, simulate, score


def print_event_log(truth, history, labels):
    print(f"{N_SCANS} scans, p_detect {P_DETECT}, {CLUTTER_RATE:.0f} false "
          f"alarms per scan on average.\n")
    for name, (first, end, _) in truth.items():
        print(f"target {name}: present scans {first}-{end - 1}")

    print("\nConfirmed tracks:")
    for tid, record in history.items():
        if "confirmed" not in record:
            continue
        name = labels[tid]
        following = f"target {name}" if name else "nothing (false track)"
        deleted = (f"deleted scan {record['deleted']:2d}"
                   if "deleted" in record else "alive at end")
        print(f"  track {tid:3d}: created scan {record['created']:2d}, "
              f"confirmed scan {record['confirmed']:2d}, {deleted}   "
              f"-> {following}")

    tentative = [r for r in history.values() if "confirmed" not in r]
    lifetimes = [r["deleted"] - r["created"] for r in tentative
                 if "deleted" in r]
    still_alive = len(tentative) - len(lifetimes)
    print(f"\n{len(history)} tracks created in all; {len(tentative)} were never "
          f"confirmed.")
    print(f"Of those, {len(lifetimes)} were deleted, after at most "
          f"{max(lifetimes)} scan(s). Still tentative when the run ended: "
          f"{still_alive}.")


def print_sweep(seeds=range(20)):
    print(f"\nThreshold sweep, {len(seeds)} seeds per row:\n")
    print("  confirm_after  delete_tentative_after  false tracks/run  "
          "extra tracks on real targets/run  scans to confirm (mean, max)")
    for confirm_after in (3, 4):
        for delete_tentative_after in (1, 2):
            false_counts, extra_counts, all_delays = [], [], []
            for seed in seeds:
                truth, _, history = simulate(seed, confirm_after,
                                             delete_tentative_after)
                false_tracks, extra_tracks, delays, _ = score(truth, history)
                false_counts.append(false_tracks)
                extra_counts.append(extra_tracks)
                all_delays += delays
            print(f"  {confirm_after:13d}  {delete_tentative_after:22d}  "
                  f"{np.mean(false_counts):16.2f}  "
                  f"{np.mean(extra_counts):32.2f}  "
                  f"{np.mean(all_delays):10.2f}, {max(all_delays):2d}")


def plot(truth, scans, history, labels):
    all_measurements = np.vstack(scans)
    plt.figure(figsize=(9, 8))
    plt.scatter(all_measurements[:, 0], all_measurements[:, 1], c="grey",
                s=8, alpha=0.35, label="all measurements (incl. clutter)")
    for name, (_, _, path) in truth.items():
        plt.plot(path[:, 0], path[:, 1], "k-", lw=2, alpha=0.35)
        plt.annotate(f"{name} enters", path[0], fontsize=9)
    plt.plot([], [], "k-", lw=2, alpha=0.35, label="truth")

    for tid, record in history.items():
        if not record["positions"]:
            continue
        xy = np.array(list(record["positions"].values()))
        style = "-" if labels[tid] else "r:"
        plt.plot(xy[:, 0], xy[:, 1], style, lw=1.8,
                 label=f"track {tid}" + ("" if labels[tid] else " (false)"))

    plt.xlim(REGION[0], REGION[1])
    plt.ylim(REGION[2], REGION[3])
    plt.legend(fontsize=8, loc="upper right")
    plt.title("Tracks started, confirmed and deleted from measurements alone\n"
              "(confirmed tracks only; every clutter point also spawned a "
              "tentative track)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("lifecycle_result.png", dpi=120)
    print("\nSaved plot -> lifecycle_result.png")


def main():
    truth, scans, history = simulate(seed=7, confirm_after=4,
                                     delete_tentative_after=1)
    _, _, _, labels = score(truth, history)
    print_event_log(truth, history, labels)
    print_sweep()
    plot(truth, scans, history, labels)


if __name__ == "__main__":
    main()
