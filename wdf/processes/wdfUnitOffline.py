"""The offline search: zero-phase conditioning, optional M4, and the segment's ridge.

`wdfUnitOffline` is `wdfUnitDSWorker` with the stages an offline analysis can
afford, each switched and set by the parameters, nothing hard-coded:

- the zero-phase band-pass (`BandPassPhase` "zero"; the Butterworth with its
  order from `AliasAttenuationDB` when `BandPassFilter` is "butterworth") and
  whichever whitening `ZeroPhaseFilter` names, "truncated" with
  `ZeroPhaseDuration` in the shipped configuration;
- **M4 glitch subtraction** (`GlitchSubtraction`,
  `wdf.processes.glitch_subtraction`): a pre-pass reads the segment, takes the
  census of its whitened stream and fits sine-Gaussian atoms to the loud
  transients, which are then subtracted from every conditioned block before
  the whitening. A seed found in coincidence with another detector's
  (`GlitchCoincidenceSeeds`, the other detectors' census files, which
  `offline_network` provides) within the light travel time plus
  `GlitchCoincidenceJitter` is protected and left in the data
  (`GlitchProtectCoincident`, on by default). The fit stretch of the noise
  model goes through the same census and is cleaned and fitted again
  (`GlitchCleanFitStretch`, default True);
- **the ridge over the segment** (`Ridge`, `wdf.analysis.ridge_tracking`):
  the whitened stream's cell map, a centred background, and `offline_ridge`
  with its backward and forward extension; the map can be taken in the
  dechirped local cosine (`RidgeDechirpAlphas`). The cells and z are written
  so that `offline_network` can combine the detectors by the minimum z.

With `GlitchSubtraction` and `Ridge` off it is `wdfUnitDSWorker`: the same
triggers from the same parameters (tested).
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

import logging
import os
import time

import numpy as np

from wdf.analysis.io import search_suffix
from wdf.processes import glitch_subtraction as gs
from wdf.processes.BandPassDownSampling import BandPassDownSampling, read_conditioned
from wdf.processes.wdfUnitDSWorker import DEFAULT_ZERO_PHASE_FILTER, wdfUnitDSWorker
from wdf.processes.zero_phase_whitening import (DEFAULT_RESPONSE_FLOOR, DEFAULT_TRUNCATION_S,
                                                MagnitudeWhitening, TruncatedWhitening)


def _array(view):
    """A view's samples in its own units (values times its scale)."""
    return np.array([view.GetY(0, i) for i in range(view.GetSize())]) * view.GetScale()


class wdfUnitOffline(wdfUnitDSWorker):
    """The offline search of one detector's segment.

    :type parameters: wdf.config.Parameters.Parameters
    :param parameters: the run configuration (`configs/offline_default.toml`
        through `load_config`).
    """

    def __init__(self, parameters):
        wdfUnitDSWorker.__init__(self, parameters)
        phase = str(getattr(self.par, "BandPassPhase", None) or "zero").lower()
        if phase != "zero":
            raise ValueError(f"the offline worker runs the zero-phase band-pass; BandPassPhase is {phase!r}")
        self.par.BandPassPhase = phase
        self.par.GlitchSubtraction = bool(getattr(self.par, "GlitchSubtraction", False))
        self.par.Ridge = bool(getattr(self.par, "Ridge", False))
        search_suffix(self.par)
        self._glitches = None
        self._settings = gs.M4Settings.from_parameters(self.par) if self.par.GlitchSubtraction else None
        from wdf.config.Parameters import reconstruction_mode

        self.reconstruction = reconstruction_mode(self.par)
        self._tracker = None
        if self.reconstruction == "track":
            if not self.par.Ridge:
                raise ValueError("Reconstruction 'track' reads the ridge's map: set Ridge = true")
            from wdf.analysis.track_reconstruction import TrackSettings

            self.par.Reconstruction = "track"
            self.track_settings = TrackSettings.from_parameters(self.par, float(self.par.resampling))

    def _observers(self, par):
        """With `Reconstruction` "track", the `TrackObserver` that keeps the
        segment's triggers as seeds."""
        if self.reconstruction != "track":
            return []
        if self._tracker is None:
            from wdf.observers.TrackObserver import TrackObserver

            self._tracker = TrackObserver(self.track_settings, float(self.par.resampling),
                                          getattr(self.par, "itf", None))
        return [self._tracker]

    # ------------------------------------------------------------ M4
    def _bank(self, whitening):
        """The sine-Gaussian bank seen through the whitening's taps."""
        taps = getattr(whitening, "taps", None)
        if taps is None:
            raise ValueError("GlitchSubtraction needs a whitening with zero-phase taps: "
                             "ZeroPhaseFilter 'truncated' or 'magnitude'")
        rate = float(self.par.resampling)
        s = self._settings
        return gs.SineGaussianBank(np.asarray(taps, dtype=float), float(whitening.sigma), rate,
                                   s.frequencies(rate), qs=s.qs, cut_tau=s.atom_cut_tau)

    def _fit_check_whitening(self, ar):
        """The zero-phase whitening of `ZeroPhaseFilter` built on `ar`."""
        zero_phase = str(getattr(self.par, "ZeroPhaseFilter", DEFAULT_ZERO_PHASE_FILTER)).lower()
        n = int(self.par.resampling)
        if zero_phase == "truncated":
            return TruncatedWhitening(ar, n, 0, duration=float(getattr(self.par, "ZeroPhaseDuration",
                                                                       DEFAULT_TRUNCATION_S)),
                                      sampling=self.par.resampling)
        if zero_phase == "magnitude":
            return MagnitudeWhitening(ar, n, 0, floor=float(getattr(self.par, "ZeroPhaseResponseFloor",
                                                                     DEFAULT_RESPONSE_FLOOR)))
        raise ValueError("GlitchSubtraction needs ZeroPhaseFilter 'truncated' or 'magnitude'")

    def _after_fit(self, learn, whiten):
        """With M4, the fit stretch's census through the model just fitted; a
        stretch holding a seed is cleaned and the model fitted again, once.
        With `GlitchRejectFitStretch` a seed left after that refuses it."""
        if not (self.par.GlitchSubtraction and bool(getattr(self.par, "GlitchCleanFitStretch", True))):
            return
        from wdf.structures.array2SeqView import array2SeqView

        self.par.GlitchCleanFitStretch = True
        rate = 1.0 / learn.GetSampling()
        start = learn.GetStart()
        x = _array(learn)

        def model_of(model):
            ar = np.array([model.ADE.GetAR(j) for j in range(self.par.ARorder + 1)])
            bank = self._bank(self._fit_check_whitening(ar))
            seeds = gs.whitened_census(x, start, rate, bank, self._settings)
            return gs.fit_seeds(x, start, rate, bank, seeds, self._settings)

        model = model_of(whiten)
        self.par.FitStretchSeeds, self.par.FitStretchAtoms = len(model.seeds), len(model)
        if not model.seeds:
            return
        x = model.subtract(x, start, rate)
        view = array2SeqView(start, rate, x.size)
        view.Fill(start, x)
        logging.info(f"Fit stretch: {len(model.seeds)} census transients, {len(model)} atoms "
                     "subtracted; the model is fitted again")
        whiten.ParametersEstimate(view.SV)
        again = model_of(whiten)
        self.par.FitStretchSeedsAfter = len(again.seeds)
        if again.seeds and bool(getattr(self.par, "GlitchRejectFitStretch", False)):
            raise ValueError(f"the fit stretch from {start} still holds {len(again.seeds)} transients "
                             "after M4; move AREstimationOffset")

    def _conditioned_segment(self, gpsStart, gpsEnd):
        """The segment's conditioned stream, read as the search reads it.

        :return: tuple -- (samples, GPS time of the first).
        """
        from py4tsa.tsa import FrameIChannel
        from py4tsa.tsa import SeqView_double_t as SV

        ds = BandPassDownSampling(self.par)
        streaming = FrameIChannel(self.par.file, self.par.channel, 1.0, gpsStart)
        data = SV()
        pieces, first = [], None
        for _ in range(int(self.par.preWhite)):
            out = read_conditioned(streaming, data, ds)
            first = out.GetStart() if first is None else first
            pieces.append(_array(out))
        streaming.SetDataLength(self.par.len)
        while True:
            out = read_conditioned(streaming, data, ds)
            first = out.GetStart() if first is None else first
            pieces.append(_array(out))
            if data.GetStart() + 2 * self.par.len > gpsEnd:
                break
        return np.concatenate(pieces), float(first)

    def _other_seeds(self):
        """The other detectors' census, from `GlitchCoincidenceSeeds`."""
        import pandas as pd

        given = getattr(self.par, "GlitchCoincidenceSeeds", None) or {}
        out = {}
        for ifo, path in dict(given).items():
            if ifo == self.par.itf:
                continue
            table = pd.read_csv(path) if os.path.getsize(path) > 1 else pd.DataFrame()
            out[ifo] = table.to_dict("records") if len(table) else []
        return out

    def _census_file(self, dir_chunk):
        return f"{dir_chunk}GlitchCensus{search_suffix(self.par)}.csv"

    def _prepare_segment(self, gpsStart, gpsEnd, whitening, dir_chunk):
        """With M4: the census, the protection of coincident seeds, the fit;
        with the ridge: the whitened stream's store."""
        import pandas as pd

        self._whitened, self._whitened_start = [], None
        if not self.par.GlitchSubtraction:
            return
        started = time.time()
        rate = float(self.par.resampling)
        x, first = self._conditioned_segment(gpsStart, gpsEnd)
        bank = self._bank(whitening)
        seeds = gs.whitened_census(x, first, rate, bank, self._settings)
        pd.DataFrame(seeds).to_csv(self._census_file(dir_chunk), index=False)
        others = self._other_seeds()
        if self._settings.protect_coincident and others:
            gs.protect_coincident({self.par.itf: seeds, **others}, self._settings.jitter_s)
            self.par.GlitchProtection = sorted(others)
        elif self._settings.protect_coincident:
            self.par.GlitchProtection = "no other detector's census given"
            logging.warning("GlitchProtectCoincident is on but no other detector's census is "
                            "given (GlitchCoincidenceSeeds): no seed can be protected")
        else:
            self.par.GlitchProtection = "off"
        self._glitches = gs.fit_seeds(x, first, rate, bank, seeds, self._settings)
        suffix = search_suffix(self.par)
        pd.DataFrame(self._glitches.table(),
                     columns=["centre", "frequency", "q", "alpha", "beta", "snr", "rate",
                              "half_length"]).to_csv(f"{dir_chunk}GlitchAtoms{suffix}.csv", index=False)
        pd.DataFrame(self._glitches.seeds).to_csv(f"{dir_chunk}GlitchSeeds{suffix}.csv", index=False)
        self.par.GlitchSeeds = len(seeds)
        self.par.GlitchSeedsProtected = int(sum(bool(s.get("protected")) for s in seeds))
        self.par.GlitchAtoms = len(self._glitches)
        self.par.GlitchFitSpan = [first, first + x.size / rate]
        logging.info(f"M4: {len(seeds)} seeds ({self.par.GlitchSeedsProtected} protected), "
                     f"{len(self._glitches)} atoms in {time.time() - started:.0f} s")

    def _read_block(self, streaming, block, downsampling):
        """The next conditioned block, with the fitted glitches subtracted."""
        out = read_conditioned(streaming, block, downsampling)
        model = self._glitches
        if model is not None and len(model):
            values = model.evaluate(out.GetStart(), 1.0 / out.GetSampling(), out.GetSize())
            if values is not None:
                scale = out.GetScale()
                for i in np.flatnonzero(values).tolist():
                    out.FillPoint(0, i, out.GetY(0, i) - float(values[i]) / scale)
        return out

    def census(self, segment):
        """The census of one segment's whitened stream, written for the other
        detectors' protection (`offline_network`).

        :type segment: tuple
        :param segment: (gpsStart, gpsEnd).
        :return: str -- the census file.
        """
        import pandas as pd

        gpsStart, gpsEnd = segment
        ID = "".join([str(self.par.channel), "_", str(int(gpsStart))])
        dir_chunk = "".join([self.par.outdir, self.par.run, "/", self.par.itf, "/", ID, '/'])
        os.makedirs(dir_chunk, exist_ok=True)
        whitening, _ = self._conditioning(gpsStart, gpsEnd, dir_chunk, ID)
        rate = float(self.par.resampling)
        x, first = self._conditioned_segment(gpsStart, gpsEnd)
        seeds = gs.whitened_census(x, first, rate, self._bank(whitening), self._settings)
        path = self._census_file(dir_chunk)
        pd.DataFrame(seeds).to_csv(path, index=False)
        return path

    # ------------------------------------------------------------ the ridge
    def _on_whitened(self, dataw):
        if self.par.Ridge:
            if self._whitened_start is None:
                self._whitened_start = dataw.GetStart()
            self._whitened.append(_array(dataw))

    def _finish_segment(self, dir_chunk):
        """The ridge over the segment's whitened stream, written beside the
        triggers."""
        if not self.par.Ridge or not self._whitened:
            return
        import pandas as pd

        from wdf.analysis.ridge_tracking import (RidgeSettings, background_z, offline_ridge,
                                                 ridge_waveform, stream_cells)

        rate = float(self.par.resampling)
        settings = RidgeSettings.from_parameters(self.par, rate)
        x = np.concatenate(self._whitened)
        X, origin, offsets, coeffs, sigmas = stream_cells(x, self._whitened_start, rate, settings)
        Z = background_z(X, settings)
        found, used = offline_ridge(Z, origin, settings, X)
        self.par.RidgeDriftUsed, self.par.RidgeXDropUsed = used.drift, used.xdrop
        top = int(getattr(self.par, "RidgeWaveforms", 10))
        self.par.RidgeWaveforms = top
        alpha = settings.alphas[0] if settings.alphas else 0.0
        starts = self._whitened_start + offsets / rate
        rows = []
        for i, c in enumerate(found):
            row = c.row()
            if i < top and not settings.alphas:
                row["waveform_energy"] = ridge_waveform(c, coeffs[alpha], sigmas[alpha], starts,
                                                        rate, origin, used)[2]
            rows.append(row)
        suffix = search_suffix(self.par)
        pd.DataFrame(rows).to_csv(f"{dir_chunk}RidgeCandidates{suffix}.csv", index=False)
        np.savez(f"{dir_chunk}RidgeCells{suffix}.npz", X=X, Z=Z, origin=origin,
                 slot=settings.slot_s, edges=settings.edges(), drift=used.drift, xdrop=used.xdrop)
        self.ridge_candidates = found
        self.par.RidgeCandidates = len(found)
        if self._tracker is not None:
            self._segment_tracks(Z, X, origin, used, coeffs[alpha], sigmas[alpha], starts, x,
                                 dir_chunk)

    def _segment_tracks(self, Z, X, origin, used, coefficients, sigmas, starts, x, dir_chunk):
        """Stage 2 of the track path over the segment: the triggers as seeds
        of tracks through its map, written beside the triggers as
        ``TrackEvents`` (CSV) and ``TrackWaveforms`` (npz)."""
        from dataclasses import replace

        from wdf.analysis.track_reconstruction import reconstruct, save_tracks

        rate = float(self.par.resampling)
        settings = replace(self.track_settings, ridge=used)
        events, waveforms, _ = reconstruct(self._tracker.records, Z, X, origin, rate, settings,
                                           getattr(self.par, "itf", None), coefficients, sigmas,
                                           starts, stream=(self._whitened_start, x))
        suffix = search_suffix(self.par)
        save_tracks(events, waveforms, f"{dir_chunk}TrackEvents{suffix}.csv",
                    f"{dir_chunk}TrackWaveforms{suffix}.npz")
        self.track_events, self.track_waveforms = events, waveforms
        self.par.TrackSeeds = len(self._tracker.records)
        self.par.TrackCount = len(events)
        self._tracker.records = []
        self._tracker.queue = []


def _aligned(maps):
    """The detectors' maps on their common slots.

    :param maps: per detector, (origin, slot, Z, X).
    :return: tuple -- (origin, list of Z, list of X).
    :raises ValueError: if the slot grids do not coincide.
    """
    slot = maps[0][1]
    first = max(m[0] for m in maps)
    last = min(m[0] + m[2].shape[0] * slot for m in maps)
    Zs, Xs = [], []
    for origin, s, Z, X in maps:
        offset = (first - origin) / slot
        if abs(s - slot) > 1e-12 or abs(offset - round(offset)) > 1e-6:
            raise ValueError("the detectors' cell maps are not on one slot grid")
        a = int(round(offset))
        n = int(round((last - first) / slot))
        Zs.append(Z[a:a + n])
        Xs.append(X[a:a + n])
    return first, Zs, Xs


def offline_network(parameters_by_ifo, segment):
    """A segment searched in several detectors, M4 protected by coincidence,
    and the network ridge.

    With `GlitchSubtraction` and `GlitchProtectCoincident`, every detector's
    census is taken first (`wdfUnitOffline.census`) and given to the others
    (`GlitchCoincidenceSeeds`), so that a seed coincident within the light
    travel time plus jitter is not subtracted anywhere. With `Ridge`, the
    detectors' z maps are combined by their minimum (`combine_min`) and the
    network ridge is written beside the first detector's output, in
    ``<outdir>/<run>/network/<gpsStart>/``.

    :type parameters_by_ifo: dict
    :param parameters_by_ifo: detector name to its run configuration.
    :type segment: tuple
    :param segment: (gpsStart, gpsEnd).
    :return: dict -- ``workers`` (per detector) and ``network`` (the network
        ridge candidates, or None).
    """
    import pandas as pd

    from wdf.analysis.ridge_tracking import RidgeSettings, combine_min, offline_ridge

    workers = {ifo: wdfUnitOffline(p) for ifo, p in parameters_by_ifo.items()}
    protect = {ifo: w.par.GlitchSubtraction and w._settings.protect_coincident
               for ifo, w in workers.items()}
    if any(protect.values()) and len(workers) > 1:
        files = {ifo: w.census(segment) for ifo, w in workers.items() if protect[ifo]}
        for ifo, w in workers.items():
            w.par.GlitchCoincidenceSeeds = {o: f for o, f in files.items() if o != ifo}
    for w in workers.values():
        w.segmentProcess(segment)
    network = None
    ridged = [w for w in workers.values() if w.par.Ridge]
    if len(ridged) > 1:
        maps = []
        for w in ridged:
            path = glob_one(w)
            z = np.load(path)
            maps.append((float(z["origin"]), float(z["slot"]), z["Z"], z["X"]))
        origin, Zs, Xs = _aligned(maps)
        first = ridged[0]
        settings = RidgeSettings.from_parameters(first.par, float(first.par.resampling))
        network, used = offline_ridge(combine_min(*Zs), origin, settings, np.minimum.reduce(Xs))
        folder = f"{first.par.outdir}{first.par.run}/network/{int(segment[0])}/"
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame([c.row() for c in network]).to_csv(
            f"{folder}NetworkRidgeCandidates{search_suffix(first.par)}.csv", index=False)
    return dict(workers=workers, network=network)


def glob_one(worker):
    """The cell map a worker wrote for its last segment."""
    import glob

    found = glob.glob(f"{worker.par.dir}RidgeCells{search_suffix(worker.par)}.npz")
    if not found:
        raise FileNotFoundError(f"no ridge cells in {worker.par.dir}")
    return found[0]
