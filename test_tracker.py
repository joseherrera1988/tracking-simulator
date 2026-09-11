"""Tests for the track lifecycle.

Run with pytest:   pytest test_tracker.py
Or just directly:  python test_tracker.py

Each rule gets a test that pins the exact scan it fires on, because an
off-by-one in a threshold doesn't crash anything -- it just makes tracks confirm
a scan late or linger a scan long, which no demo would visibly catch. The
counting convention is track.py's: the measurement that creates a track is its
first hit.
"""

import numpy as np

from tracker import Tracker

HERE = np.array([[0.0, 0.0]])
NOTHING = np.empty((0, 2))


def _events(tracker, scans):
    """Run scans through the tracker and collect (scan, event, id) triples."""
    log = []
    for k, scan in enumerate(scans):
        log += [(k, event, tid) for event, tid in tracker.step(scan)]
    return log


def test_an_unclaimed_measurement_starts_a_tentative_track():
    tracker = Tracker()
    events = tracker.step(HERE)

    assert events == [("created", 1)], f"got {events}"
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].status == "tentative"
    assert tracker.confirmed_tracks == []


def test_confirmed_on_the_scan_it_reaches_confirm_after_hits():
    """confirm_after=3: created on scan 0 (hit 1), hit 2 on scan 1, confirmed on
    scan 2 when the third hit lands -- not a scan before or after."""
    tracker = Tracker(confirm_after=3)
    events = _events(tracker, [HERE, HERE, HERE, HERE])

    assert events == [(0, "created", 1), (2, "confirmed", 1)], f"got {events}"
    assert [t.id for t in tracker.confirmed_tracks] == [1]


def test_tentative_track_dies_on_first_miss_by_default():
    tracker = Tracker()
    events = _events(tracker, [HERE, HERE, NOTHING])

    assert events == [(0, "created", 1), (2, "deleted", 1)], f"got {events}"
    assert tracker.tracks == []


def test_delete_tentative_after_lets_a_new_track_survive_a_miss():
    """The knob that trades clutter cleanup against restarting real tracks:
    at 2, one missed scan is forgiven; two in a row are not."""
    tracker = Tracker(confirm_after=5, delete_tentative_after=2)
    events = _events(tracker, [HERE, NOTHING, HERE, NOTHING, NOTHING])

    assert events == [(0, "created", 1), (4, "deleted", 1)], f"got {events}"


def test_confirmed_track_survives_until_delete_confirmed_after_misses():
    """Confirmed on scan 2, then nothing: misses 1-3 are tolerated and the
    fourth, on scan 6, deletes it."""
    tracker = Tracker(confirm_after=3, delete_confirmed_after=4)
    events = _events(tracker, [HERE, HERE, HERE] + [NOTHING] * 5)

    assert events == [(0, "created", 1), (2, "confirmed", 1),
                      (6, "deleted", 1)], f"got {events}"


def test_isolated_clutter_never_confirms():
    """A point that never repeats near itself spawns a tentative track that
    dies the next scan. Far apart, so no point lands in an earlier one's gate."""
    tracker = Tracker()
    scans = [np.array([[1000.0 * k, 0.0]]) for k in range(6)]
    events = _events(tracker, scans)

    assert not any(event == "confirmed" for _, event, _ in events), events
    created = [tid for _, event, tid in events if event == "created"]
    deleted = [tid for _, event, tid in events if event == "deleted"]
    assert created == [1, 2, 3, 4, 5, 6]
    assert deleted == [1, 2, 3, 4, 5], "each should die the scan after it began"


def test_ids_are_never_reused():
    """A deleted track's ID stays retired, so a log line can never be confused
    with a later, unrelated track."""
    tracker = Tracker()
    _events(tracker, [HERE, NOTHING, HERE])

    assert [t.id for t in tracker.tracks] == [2]


def test_a_confirmed_track_keeps_a_measurement_a_tentative_track_wants():
    """The failure lifecycle_demo.py showed: a newer tentative track sits
    closer to a measurement than the confirmed track that needs it.

    Track 1 is confirmed at the origin on scan 1, and that scan's second point,
    at (6, 0), starts tentative track 2. On scan 2 a single measurement lands at
    (5, 0): far nearer track 2's prediction, but inside track 1's gate. In one
    joint assignment track 2 takes it. With confirmed tracks choosing first,
    track 1 does, and track 2 -- having missed -- is deleted.
    """
    tracker = Tracker(confirm_after=2)
    tracker.step(HERE)
    tracker.step(np.array([[0.0, 0.0], [6.0, 0.0]]))
    assert [(t.id, t.status) for t in tracker.tracks] == [
        (1, "confirmed"), (2, "tentative")]

    events = tracker.step(np.array([[5.0, 0.0]]))

    assert events == [("deleted", 2)], f"got {events}"
    assert [t.id for t in tracker.tracks] == [1]
    assert tracker.tracks[0].misses == 0, "track 1 should have been updated"


def test_an_empty_scan_on_an_empty_tracker():
    tracker = Tracker()
    assert tracker.step(NOTHING) == []
    assert tracker.tracks == []


if __name__ == "__main__":
    # lets you run the file directly without pytest installed
    test_an_unclaimed_measurement_starts_a_tentative_track()
    test_confirmed_on_the_scan_it_reaches_confirm_after_hits()
    test_tentative_track_dies_on_first_miss_by_default()
    test_delete_tentative_after_lets_a_new_track_survive_a_miss()
    test_confirmed_track_survives_until_delete_confirmed_after_misses()
    test_isolated_clutter_never_confirms()
    test_ids_are_never_reused()
    test_a_confirmed_track_keeps_a_measurement_a_tentative_track_wants()
    test_an_empty_scan_on_an_empty_tracker()
    print("all tests passed")
