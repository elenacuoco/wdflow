"""Seeds for the track reconstruction, taken from the search as it runs.

`TrackObserver` is registered on the `ParameterEstimation` of each search when
`Reconstruction` is "track". It keeps every trigger's record with its
coefficients, which the track needs for its seed's loudest tile.

Online (`wdfUnitOnline`) the worker calls `drain` after every block: the
seeds whose slice of the map (`TrackHorizon` past their window included) is
complete are reconstructed from the map rows and window coefficients the
streaming map still holds, and leave the queue. A track of one drain may share
cells of the map with a track emitted by an earlier drain (a long inspiral is
seeded again and again as it rises): `TrackOverlapPolicy` "drop" discards it
and counts its seeds as absorbed; "supersede" emits it when its score is
higher than that of every emitted track it overlaps, with their seeds and
their ids in `track_supersedes`, and discards it otherwise. An emitted row is
never changed afterwards; the superseded rows stay in the output and are
skipped by `load_tracks` and the coincidence. Offline (`wdfUnitOffline`) the
records are read once the segment's map is computed.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

import numpy as np

from wdf.observers.observer import Observer


class TrackObserver(Observer):
    """Queues the triggers, with their coefficients, as track seeds.

    :type settings: wdf.analysis.track_reconstruction.TrackSettings
    :param settings: the reconstruction's settings.
    :type fs: float
    :param fs: sampling rate of the searched stream, Hz.
    :type ifo: str
    :param ifo: the detector, written in the track events.
    """

    def __init__(self, settings, fs, ifo=None):
        Observer.__init__(self)
        self.settings, self.fs, self.ifo = settings, float(fs), ifo
        self.queue = []
        self.records = []
        #: The last emitted tracks still in force: ``(cluster_id, cells,
        #: score, seeds)``; a superseding track takes the cells of the tracks
        #: it replaces, so that later tracks are compared with it.
        self.emitted = []
        self.count = 0
        self.dropped = 0
        #: Seeds of the tracks discarded for overlapping an emitted track.
        self.absorbed = 0
        #: Tracks superseded, and the seeds carried over from them.
        self.superseded = 0
        self.carried = 0

    def update(self, event):
        """Keep one trigger's record, coefficients included."""
        record = event.record()
        self.queue.append(record)
        self.records.append(record)

    def drain(self, origin, next_slot, rows, coefficients, settings):
        """Reconstruct the seeds whose slice of the map is complete.

        :type origin: float
        :param origin: GPS start of the map's slot 0.
        :type next_slot: int
        :param next_slot: slots of the map emitted so far.
        :type rows: list
        :param rows: the map rows held, ``(slot, z, cells)`` in slot order.
        :type coefficients: tuple
        :param coefficients: the held windows (GPS starts, coefficients,
            scales), as `StreamingCells.held` returns them.
        :type settings: TrackSettings or None
        :param settings: the settings with the calibrated drift and X-drop;
            None while the ridge is not calibrated, when the ready seeds are
            dropped.
        :return: tuple -- (list of event rows, ``{cluster_id: (t0,
            waveform)}``).
        """
        from wdf.analysis.track_reconstruction import (reconstruct, seed_window, track_row)

        ready, waiting = [], []
        for record in self.queue:
            seed = seed_window(record, self.settings, origin)
            (ready if seed.slot_hi <= next_slot else waiting).append(record)
        self.queue = waiting
        if not ready:
            return [], {}
        if settings is None or not rows:
            self.dropped += len(ready)
            return [], {}
        first = rows[0][0]
        Z = np.array([r[1] for r in rows])
        X = np.array([r[2] for r in rows])
        starts, coeffs, sigmas = coefficients
        events, waveforms, tracks = reconstruct(ready, Z, X, origin, self.fs, settings, self.ifo,
                                                coeffs, sigmas, starts, first_slot=first,
                                                first_id=self.count)
        out, kept = [], {}
        policy = self.settings.overlap_policy
        for (_, row), track in zip(events.iterrows(), tracks):
            cells = set(zip(track.slots.tolist(), track.states.tolist()))
            overlapped = [e for e in self.emitted if cells & e[1]]
            cid = self.count
            if overlapped:
                if policy != "supersede" or track.score <= max(e[2] for e in overlapped):
                    self.absorbed += len(track.seeds)
                    continue
                # The new track replaces the emitted ones it overlaps: their
                # seeds become its own, their cells block later tracks as its.
                for e in overlapped:
                    track.seeds.extend(s for s in e[3] if not any(s is q for q in track.seeds))
                    cells |= e[1]
                    self.emitted.remove(e)
                self.superseded += len(overlapped)
                self.carried += sum(len(e[3]) for e in overlapped)
                row = track_row(track, cid, self.ifo, self.fs, self.settings.event_instant,
                                supersedes=[e[0] for e in overlapped])
            else:
                row = row.to_dict()
            self.emitted.append((cid, cells, float(track.score), list(track.seeds)))
            row["cluster_id"] = cid
            row["seed_end"] = max(s.t_hi for s in track.seeds)
            if track.waveform is not None:
                kept[cid] = (float(track.waveform[0]), np.asarray(track.waveform[1]))
            self.count += 1
            out.append(row)
        self.emitted = self.emitted[-256:]
        return out, kept
