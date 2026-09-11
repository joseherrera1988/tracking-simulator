"""Many tracks at once: when to start one, when to trust it, when to drop it.

Association (association.py) decides which measurement goes with which EXISTING
track. It has nothing to say about measurements no track claims, or about tracks
that stop getting measurements. Once the sensor misses targets and reports
clutter, both happen every scan, and something has to decide what they mean.
That's the track lifecycle, and the Tracker owns it:

  - INITIATION: every measurement no track claims starts a new tentative track.
    Most of these are clutter, and that's expected -- see below.
  - CONFIRMATION: a tentative track that collects `confirm_after` hits is
    promoted to confirmed. Only confirmed tracks are reported as targets.
  - DELETION: a track that misses `delete_confirmed_after` scans in a row is
    deleted if confirmed, `delete_tentative_after` if still tentative.

WHY INITIATE FROM A SINGLE MEASUREMENT. The alternative is to hold unmatched
measurements in a separate pending list and only start a track once one repeats
near where it was. That delays committing to a track, but it needs a second data
structure with its own lifecycle to test. Starting a tentative track immediately
reuses the machinery that already exists: an isolated clutter point almost never
repeats near its own prediction, so the track it spawns misses on the very next
scan and is deleted. The cost is transient objects -- the track list briefly
carries one tentative track per clutter point.

THE STRICTEST KNOB is delete_tentative_after. At 1, a tentative track dies on its
first miss, which kills clutter fast but also kills a REAL target's track if the
sensor happens to miss it during its first few scans; the track then re-forms
from the next detection under a new ID. Raising it trades faster clutter cleanup
for fewer restarts, and lifecycle_demo.py shows both sides.

KNOWN GAP: association treats tentative and confirmed tracks alike, so a
clutter-spawned tentative track can compete with a confirmed track for a
measurement in the assignment and occasionally win it. Real systems give
confirmed tracks priority. Not built until a demo shows it happening -- but if a
confirmed track inexplicably drops a measurement, this is the first suspect.
"""

from association import associate
from gating import CHI2_99_2D
from track import Track


class Tracker:
    def __init__(self, confirm_after=3, delete_tentative_after=1,
                 delete_confirmed_after=5, dt=1.0, process_var=1.0,
                 meas_var=25.0, gate=CHI2_99_2D):
        """
        Args:
            confirm_after:          hits needed to confirm a tentative track,
                                    counting the measurement that created it.
            delete_tentative_after: consecutive misses that delete a tentative
                                    track.
            delete_confirmed_after: consecutive misses that delete a confirmed
                                    track.
            dt, process_var, meas_var: passed to every Track this creates.
            gate:                   association threshold, squared Mahalanobis.
        """
        self.confirm_after = confirm_after
        self.delete_tentative_after = delete_tentative_after
        self.delete_confirmed_after = delete_confirmed_after
        self.track_kwargs = dict(dt=dt, process_var=process_var,
                                 meas_var=meas_var)
        self.gate = gate

        self.tracks = []
        self.next_id = 1

    def step(self, measurements):
        """Run one scan.

        Args:
            measurements: (n, 2) array of this scan's reports, possibly empty.

        Returns:
            List of (event, track_id) pairs, where event is "created",
            "confirmed" or "deleted", in the order they happened this scan.
        """
        events = []

        # 1. predict, associate, correct -- the Phase 3 scan, unchanged
        for track in self.tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_measurements = associate(
            self.tracks, measurements, threshold=self.gate
        )
        for track_index, measurement_index in matches:
            self.tracks[track_index].correct(measurements[measurement_index])
        for track_index in unmatched_tracks:
            self.tracks[track_index].correct(None)

        # 2. promote tentative tracks with enough hits
        for track in self.tracks:
            if track.status == "tentative" and track.hits >= self.confirm_after:
                track.status = "confirmed"
                events.append(("confirmed", track.id))

        # 3. delete tracks that have missed too many scans in a row
        survivors = []
        for track in self.tracks:
            if track.status == "tentative":
                limit = self.delete_tentative_after
            else:
                limit = self.delete_confirmed_after
            if track.misses >= limit:
                events.append(("deleted", track.id))
            else:
                survivors.append(track)
        self.tracks = survivors

        # 4. every measurement no track claimed starts a tentative track. These
        # are created after association, so they sit out this scan's predict
        # and first face a measurement on the next one.
        for measurement_index in unmatched_measurements:
            track = Track(measurements[measurement_index],
                          track_id=self.next_id, **self.track_kwargs)
            self.next_id += 1
            self.tracks.append(track)
            events.append(("created", track.id))

        return events

    @property
    def confirmed_tracks(self):
        return [t for t in self.tracks if t.status == "confirmed"]
