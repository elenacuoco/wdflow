"""Waveform completion: the whole transient under a detected event, holes
between its tiles filled, in the time domain.

The wavegram reconstruction (`reconstruction.stitch`, `event_waveform`)
inverts the coefficients that survived the detection threshold, and nothing
else: along a chirp the tiles that fell below it leave holes in the waveform,
and what a parameter estimation receives is a series with gaps. The track
waveform (`track_reconstruction.track_waveform`) is continuous but limited
to the ridge's band and to the track's span.

This stage runs after the detection and before the parameter estimation. It
builds a time-frequency **mask** around the event -- the union of its tiles,
the holes between them bridged, the whole widened in time and in octaves, and
the track's band when a track exists -- and inverts **every** coefficient
inside the mask, with no detection threshold, over a padded window of the
zero-phase whitened stream, in one basis (`CompletionBasis`): the cosine-packet
best basis of the piece, a local cosine basis, or a p4TSA wavelet basis. The
coefficients outside the mask are zeroed, so what fills the holes is the data
there, read in the basis, not a model of the transient. The mask carries no
assumption on the transient's shape: the bridging interpolates the band of the
tiles in log-frequency across a hole, symmetric in time (a rising and a falling
sweep are treated alike), and every width is a parameter.

`CompletionMethod = "mp"` instead fits sine-Gaussian atoms
(`glitch_subtraction.SineGaussianBank`, the M4 dictionary) inside the mask by
matching pursuit on the whitened stream, and returns their sum.

The output (`CompletedWaveform`) is the completed whitened waveform on the
stream's own sampling, NaN-free, its strain version through the zero-phase
whitening of the model in force (`hrss.strain_from_whitened`), the mask, a
coverage number (the fraction of a reference ridge -- the track, or the
injected one -- that the mask holds) and the overlap with the sparse
reconstruction.

The completion runs only for events whose tiles cover their own span poorly:
`tile_coverage`, the fraction of the event's span (its tiles bridged in time
and filled in frequency) that the tiles occupy, read from the tiles alone,
is compared with
`CompletionMaxTileCoverage`; above it the sparse reconstruction is returned
unchanged, flagged ``completed=False``.

Nothing calls this stage by default; a run's outputs are unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from wdf.analysis.wavelets import LOCAL_COSINE
from wdf.config.constants import MAD_TO_SIGMA
from wdf.config.defaults import defaults

COSINE_PACKETS = "CosinePackets"
METHODS = ("mask", "mp")
TILE_BANDS = ("extent", "centroid")
PACKET_COSTS = ("l1", "entropy", "wdfUniversal", "wdfBlock")
#: Table ``[waveform_completion]`` of ``wdf/config/defaults.toml``.
_WC = defaults("waveform_completion")
#: Smallest frequency a log2 is taken of, Hz: a floor against log(0).
_LOG_FLOOR_HZ = _WC["log_floor_hz"]


# ================================================================ defaults
#: The defaults of the completion: table ``[waveform_completion]`` of
#: ``wdf/config/defaults.toml``, keyed by parameter name. A key the run does
#: not set takes its value here. "CompletionSlot" unset: `CompletionSlotSamples`
#: over the stream's rate; "CompletionLowFrequency" unset: the run's
#: `MinFrequency` (0 without one).
DEFAULTS = {k: v for k, v in _WC.items() if k[0].isupper()}


# ================================================================ settings
@dataclass
class CompletionSettings:
    """Every choice of the completion, read from the run parameters; a key
    the run does not set takes its value in `DEFAULTS`.

    :param method: `CompletionMethod`: "mask" inverts every coefficient
        inside the mask; "mp" fits sine-Gaussian atoms inside it.
    :param basis: `CompletionBasis`: ``CosinePackets<m>`` (the best basis of
        the piece), ``LocalCos<M>``, or a p4TSA wavelet name (a mother, or a
        mother with ``P<depth>``).
    :param packet_cost: `CompletionPacketCost`, the cosine-packet cost.
    :param slot_s: `CompletionSlot`, s: the mask's time cell; when not set,
        `CompletionSlotSamples` samples at the stream's rate.
    :param octave_step: `CompletionOctaveStep`: the mask's frequency cell,
        octaves.
    :param dilation_time: `CompletionDilationTime`, s: the mask grows by this
        on each side in time.
    :param dilation_octaves: `CompletionDilationOctaves`: ... and by this in
        frequency, octaves, on each side.
    :param bridge_gap: `CompletionBridgeGap`, s: a hole between two occupied
        time cells shorter than this is filled with the band interpolated in
        log-frequency between its two sides; 0 bridges nothing.
    :param tile_band: `CompletionTileBand`: "extent" marks each tile's whole
        band; "centroid" marks, in each time cell, the energy-weighted
        log-frequency centroid of the tiles there, `band` octaves on each
        side (a compact mask when the tiles are wide, as a pyramid's are).
    :param band: `CompletionBand`, octaves: half-width of the band around a
        track's ridge, and around the tiles' centroid with "centroid".
    :param threshold: `CompletionThreshold`: soft threshold, in units of the
        piece's noise scale, applied to the kept coefficients; 0 keeps them
        as they are.
    :param pad: `CompletionPad`, s: the waveform extends this much before the
        event's first tile and after its last.
    :param low_frequency: `CompletionLowFrequency`, Hz: the mask's floor;
        when not set, the run's `MinFrequency`.
    :param stop_snr: `CompletionStopSNR`: "mp" stops when the best atom's SNR
        is below it.
    :param max_atoms: `CompletionMaxAtoms`: ... or after this many atoms.
    :param qs: `CompletionQs`: quality factors of the atoms.
    :param frequency_count: `CompletionFrequencyCount`: atom frequencies,
        log-spaced over the mask's band.
    :param max_tile_coverage: `CompletionMaxTileCoverage`: the completion
        runs when the event's tile coverage (`tile_coverage`) is at or below
        this; above it the sparse reconstruction is returned as it is. 1
        completes every event.
    :param margin_segments: `CompletionMarginSegments`: the piece holds this
        many segments of a ``CosinePackets<m>`` or ``LocalCos<M>`` basis
        beyond the mask on each side.
    :param wavelet_margin: `CompletionWaveletMargin`, s: ... and this much
        for a p4TSA wavelet.
    :param slot_samples: `CompletionSlotSamples`: `slot_s` in samples at the
        stream's rate when `CompletionSlot` is not set.
    """

    method: str
    basis: str
    packet_cost: str
    slot_s: float
    octave_step: float
    dilation_time: float
    dilation_octaves: float
    bridge_gap: float
    tile_band: str
    band: float
    threshold: float
    pad: float
    low_frequency: float
    stop_snr: float
    max_atoms: int
    qs: tuple
    frequency_count: int
    max_tile_coverage: float
    margin_segments: int
    wavelet_margin: float
    slot_samples: int

    KEYS = dict(method="CompletionMethod", basis="CompletionBasis",
                packet_cost="CompletionPacketCost", slot_s="CompletionSlot",
                octave_step="CompletionOctaveStep", dilation_time="CompletionDilationTime",
                dilation_octaves="CompletionDilationOctaves", bridge_gap="CompletionBridgeGap",
                tile_band="CompletionTileBand", band="CompletionBand",
                threshold="CompletionThreshold", pad="CompletionPad",
                low_frequency="CompletionLowFrequency", stop_snr="CompletionStopSNR",
                max_atoms="CompletionMaxAtoms", qs="CompletionQs",
                frequency_count="CompletionFrequencyCount",
                max_tile_coverage="CompletionMaxTileCoverage",
                margin_segments="CompletionMarginSegments",
                wavelet_margin="CompletionWaveletMargin",
                slot_samples="CompletionSlotSamples")
    #: How each value is read.
    TYPES = dict(method=str, basis=str, packet_cost=str, slot_s=float, octave_step=float,
                 dilation_time=float, dilation_octaves=float, bridge_gap=float, tile_band=str,
                 band=float, threshold=float, pad=float, low_frequency=float, stop_snr=float,
                 max_atoms=int, qs=lambda v: tuple(float(q) for q in v), frequency_count=int,
                 max_tile_coverage=float, margin_segments=int, wavelet_margin=float,
                 slot_samples=int)

    @classmethod
    def from_parameters(cls, par, rate, defaults=None):
        """The settings a run asks for, defaults filled in and written back.

        :param par: run configuration; it receives every value used.
        :type rate: float
        :param rate: sampling rate of the whitened stream, Hz.
        :type defaults: dict or None
        :param defaults: the values of the keys `par` does not set;
            `DEFAULTS` when None.
        :return: CompletionSettings
        :raises ValueError: on a method, basis or cost that is not known, or
            a width that is negative.
        """
        d = DEFAULTS if defaults is None else defaults

        def get(key):
            value = getattr(par, key, None)
            return d.get(key) if value is None else value

        values = {name: get(key) for name, key in cls.KEYS.items()}
        if values["slot_s"] is None:
            values["slot_s"] = int(values["slot_samples"]) / float(rate)
        if values["low_frequency"] is None:
            low = getattr(par, "MinFrequency", None)
            values["low_frequency"] = float(low) if low else 0.0
        out = cls(**{name: cls.TYPES[name](v) for name, v in values.items()})
        out.check(rate)
        for name, key in cls.KEYS.items():
            value = getattr(out, name)
            setattr(par, key, list(value) if isinstance(value, tuple) else value)
        return out

    @classmethod
    def defaults(cls, rate, **values):
        """The settings with every key at its default but those given by
        their field name (`KEYS`), at the stream rate `rate`."""
        from types import SimpleNamespace

        return cls.from_parameters(SimpleNamespace(**{cls.KEYS[k]: v for k, v in values.items()}), rate)

    def check(self, rate=None):
        """Refuse settings the completion cannot run."""
        if self.method not in METHODS:
            raise ValueError(f"CompletionMethod {self.method!r}: one of {METHODS}")
        if self.packet_cost not in PACKET_COSTS:
            raise ValueError(f"CompletionPacketCost {self.packet_cost!r}: one of {PACKET_COSTS}")
        basis_kind(self.basis)
        if self.tile_band not in TILE_BANDS:
            raise ValueError(f"CompletionTileBand {self.tile_band!r}: one of {TILE_BANDS}")
        for name in ("slot_s", "octave_step"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{self.KEYS[name]} must be positive")
        for name in ("dilation_time", "dilation_octaves", "bridge_gap", "band", "threshold",
                     "pad", "low_frequency", "stop_snr"):
            if getattr(self, name) < 0:
                raise ValueError(f"{self.KEYS[name]} must not be negative")
        if not 0.0 <= self.max_tile_coverage <= 1.0:
            raise ValueError("CompletionMaxTileCoverage must be between 0 and 1")
        if self.margin_segments < 0 or self.wavelet_margin < 0:
            raise ValueError("CompletionMarginSegments and CompletionWaveletMargin must not be negative")
        if self.max_atoms < 1 or self.frequency_count < 1:
            raise ValueError("CompletionMaxAtoms and CompletionFrequencyCount must be at least 1")
        if not self.qs or min(self.qs) <= 0:
            raise ValueError("CompletionQs must be positive quality factors")
        if rate is not None and self.low_frequency >= 0.5 * float(rate):
            raise ValueError(f"CompletionLowFrequency {self.low_frequency} Hz is at or above Nyquist")

    def record(self):
        """The settings as a dict."""
        return asdict(self)


def basis_kind(basis):
    """"packets", "localcos" or "wavelet" for a basis name.

    :raises ValueError: on a name that is none of them.
    """
    name = str(basis)
    if name.startswith(COSINE_PACKETS):
        packet_segment(name)
        return "packets"
    if name.startswith(LOCAL_COSINE):
        from wdf.analysis.wavelets import _local_cosine_segment
        _local_cosine_segment(name)
        return "localcos"
    from wdf.analysis.wavelets import mother_wavelet, packet_depth
    packet_depth(name)
    mother = mother_wavelet(name)
    try:
        from py4tsa.tsa import WaveletTransform
    except ImportError:                     # checked when the transform runs
        return "wavelet"
    if not mother or not hasattr(WaveletTransform, mother):
        raise ValueError(f"CompletionBasis {name!r}: {mother!r} is not a py4tsa wavelet")
    return "wavelet"


def packet_segment(basis):
    """The smallest segment ``m`` of a ``CosinePackets<m>`` name."""
    digits = str(basis)[len(COSINE_PACKETS):]
    if not digits.isdigit() or int(digits) < 2 or int(digits) & (int(digits) - 1):
        raise ValueError(f"CompletionBasis {basis!r}: CosinePackets takes a power of 2, "
                         "as CosinePackets32")
    return int(digits)


# ================================================================ the mask
@dataclass
class CompletionMask:
    """A time-frequency mask on a grid of time cells by log-frequency cells.

    :param t0: GPS time of the first time cell's start.
    :param slot_s: width of a time cell, s.
    :param log_edges: log2 of the frequency cells' edges, Hz, ascending
        (``n_bins + 1`` values).
    :param cells: boolean ``(n_slots, n_bins)``.
    """

    t0: float
    slot_s: float
    log_edges: np.ndarray
    cells: np.ndarray

    @classmethod
    def empty(cls, t_start, t_end, f_low, f_high, slot_s, octave_step):
        """An empty mask over ``[t_start, t_end)`` and ``[f_low, f_high]``."""
        f_low = max(float(f_low), _LOG_FLOOR_HZ)
        n_bins = max(int(np.ceil((np.log2(f_high) - np.log2(f_low)) / octave_step)), 1)
        log_edges = np.log2(f_low) + np.arange(n_bins + 1) * float(octave_step)
        n_slots = max(int(np.ceil((t_end - t_start) / slot_s)), 1)
        return cls(float(t_start), float(slot_s), log_edges, np.zeros((n_slots, n_bins), bool))

    # ---- geometry
    @property
    def n_slots(self):
        return self.cells.shape[0]

    @property
    def n_bins(self):
        return self.cells.shape[1]

    @property
    def t_end(self):
        return self.t0 + self.n_slots * self.slot_s

    @property
    def f_low(self):
        return float(2.0 ** self.log_edges[0])

    @property
    def f_high(self):
        return float(2.0 ** self.log_edges[-1])

    def slot_range(self, t_lo, t_hi):
        """Cells ``[a, b)`` meeting ``[t_lo, t_hi)``; empty when none."""
        a = int(np.floor((t_lo - self.t0) / self.slot_s + 1e-9))
        b = int(np.ceil((t_hi - self.t0) / self.slot_s - 1e-9))
        return max(a, 0), min(max(b, a + 1), self.n_slots)

    def bin_range(self, f_lo, f_hi):
        """Cells ``[a, b)`` meeting ``[f_lo, f_hi)``; empty when none."""
        step = self.log_edges[1] - self.log_edges[0]
        lo = np.log2(max(float(f_lo), _LOG_FLOOR_HZ))
        hi = np.log2(max(float(f_hi), _LOG_FLOOR_HZ))
        a = int(np.floor((lo - self.log_edges[0]) / step + 1e-9))
        b = int(np.ceil((hi - self.log_edges[0]) / step - 1e-9))
        return max(a, 0), min(max(b, a + 1), self.n_bins)

    # ---- building
    def add_boxes(self, t_lo, t_hi, f_lo, f_hi):
        """Mark the cells meeting each box ``[t_lo, t_hi) x [f_lo, f_hi)``."""
        for a, b, c, d in zip(np.atleast_1d(t_lo), np.atleast_1d(t_hi),
                              np.atleast_1d(f_lo), np.atleast_1d(f_hi)):
            s0, s1 = self.slot_range(a, b)
            b0, b1 = self.bin_range(c, d)
            if s1 > s0 and b1 > b0:
                self.cells[s0:s1, b0:b1] = True
        return self

    def add_centroids(self, t_lo, t_hi, f_lo, f_hi, energy, band):
        """Mark, in each time cell some tile meets, the energy-weighted
        log-frequency centroid of those tiles (each tile's centre, weighted by
        its energy over the cells it spans) `band` octaves on each side."""
        t_lo, t_hi = np.atleast_1d(t_lo).astype(float), np.atleast_1d(t_hi).astype(float)
        f_lo, f_hi = np.atleast_1d(f_lo).astype(float), np.atleast_1d(f_hi).astype(float)
        energy = np.atleast_1d(energy).astype(float)
        weight = np.zeros(self.n_slots)
        moment = np.zeros(self.n_slots)
        for a, b, c, d, e in zip(t_lo, t_hi, f_lo, f_hi, energy):
            s0, s1 = self.slot_range(a, b)
            if s1 <= s0 or not (d > 0 and e > 0):
                continue
            centre = 0.5 * (np.log2(max(c, _LOG_FLOOR_HZ)) + np.log2(d))
            w = e / (s1 - s0)
            weight[s0:s1] += w
            moment[s0:s1] += w * centre
        for s in np.flatnonzero(weight > 0):
            here = moment[s] / weight[s]
            b0, b1 = self.bin_range(2.0 ** (here - band), 2.0 ** (here + band))
            if b1 > b0:
                self.cells[s, b0:b1] = True
        return self

    def add_ridge(self, times, freqs, band):
        """Mark ``f / 2**band .. f * 2**band`` at each point of a ridge, the
        points joined slot by slot (the ridge interpolated in log-frequency)."""
        t = np.asarray(times, dtype=float)
        f = np.asarray(freqs, dtype=float)
        ok = np.isfinite(t) & np.isfinite(f) & (f > 0)
        t, f = t[ok], f[ok]
        if not t.size:
            return self
        order = np.argsort(t)
        t, lf = t[order], np.log2(f[order])
        s0, s1 = self.slot_range(t[0], t[-1] + 1e-9)
        for s in range(s0, s1):
            centre = self.t0 + (s + 0.5) * self.slot_s
            here = np.interp(np.clip(centre, t[0], t[-1]), t, lf)
            b0, b1 = self.bin_range(2.0 ** (here - band), 2.0 ** (here + band))
            if b1 > b0:
                self.cells[s, b0:b1] = True
        return self

    def bridge(self, max_gap_s):
        """Fill every run of empty time cells shorter than `max_gap_s` that
        sits between two occupied ones with the band interpolated, in
        log-frequency, between the band on its left and the band on its
        right."""
        if max_gap_s <= 0:
            return self
        occupied = self.cells.any(axis=1)
        idx = np.flatnonzero(occupied)
        if idx.size < 2:
            return self
        limit = int(np.floor(max_gap_s / self.slot_s + 1e-9))
        for left, right in zip(idx[:-1], idx[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > limit:
                continue
            lo_l, hi_l = _band_of(self.cells[left])
            lo_r, hi_r = _band_of(self.cells[right])
            for k, s in enumerate(range(left + 1, right)):
                w = (k + 1) / (gap + 1)
                lo = int(round((1 - w) * lo_l + w * lo_r))
                hi = int(round((1 - w) * hi_l + w * hi_r))
                self.cells[s, lo:hi + 1] = True
        return self

    def dilate(self, time_s, octaves):
        """Grow the mask by `time_s` on each side in time and by `octaves` on
        each side in frequency."""
        from scipy.ndimage import binary_dilation

        step = self.log_edges[1] - self.log_edges[0]
        nt = int(np.ceil(time_s / self.slot_s - 1e-9)) if time_s > 0 else 0
        nf = int(np.ceil(octaves / step - 1e-9)) if octaves > 0 else 0
        if nt == 0 and nf == 0:
            return self
        structure = np.ones((2 * nt + 1, 2 * nf + 1), bool)
        self.cells = binary_dilation(self.cells, structure=structure)
        return self

    # ---- reading
    def boxes(self):
        """The occupied cells as boxes ``(t_lo, t_hi, f_lo, f_hi)``, one per
        maximal run of occupied frequency cells in a time cell."""
        out = []
        f = 2.0 ** self.log_edges
        for s in range(self.n_slots):
            row = self.cells[s]
            if not row.any():
                continue
            edges = np.diff(np.concatenate(([0], row.astype(int), [0])))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                out.append((self.t0 + s * self.slot_s, self.t0 + (s + 1) * self.slot_s, f[a], f[b]))
        return out

    def contains(self, times, freqs):
        """Whether each point ``(time, frequency)`` falls in an occupied cell."""
        t = np.asarray(times, dtype=float)
        f = np.asarray(freqs, dtype=float)
        s = np.floor((t - self.t0) / self.slot_s).astype(int)
        step = self.log_edges[1] - self.log_edges[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.floor((np.log2(np.where(f > 0, f, np.nan)) - self.log_edges[0]) / step)
        ok = (s >= 0) & (s < self.n_slots) & np.isfinite(b) & (b >= 0) & (b < self.n_bins)
        out = np.zeros(t.shape, bool)
        out[ok] = self.cells[s[ok], b[ok].astype(int)]
        return out

    def spanned(self):
        """The cells from the lowest occupied one to the highest in each time
        cell: the mask with its holes in frequency filled."""
        out = self.cells.copy()
        rows = np.flatnonzero(out.any(axis=1))
        for s in rows:
            idx = np.flatnonzero(out[s])
            out[s, idx[0]:idx[-1] + 1] = True
        return out

    def coverage(self, times, freqs):
        """The fraction of a ridge's points the mask holds, NaN without
        points."""
        inside = self.contains(times, freqs)
        return float(inside.mean()) if inside.size else float("nan")

    def keeps(self, t_lo, t_hi, f_lo, f_hi):
        """Whether each tile ``[t_lo, t_hi) x [f_lo, f_hi)`` meets an occupied
        cell; a summed-area table makes it one lookup per tile."""
        t_lo, t_hi = np.asarray(t_lo, dtype=float), np.asarray(t_hi, dtype=float)
        f_lo, f_hi = np.asarray(f_lo, dtype=float), np.asarray(f_hi, dtype=float)
        n = t_lo.size
        step = self.log_edges[1] - self.log_edges[0]
        s0 = np.clip(np.floor((t_lo - self.t0) / self.slot_s + 1e-9), 0, self.n_slots).astype(int)
        s1 = np.clip(np.ceil((t_hi - self.t0) / self.slot_s - 1e-9), 0, self.n_slots).astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            lo = np.log2(np.maximum(f_lo, _LOG_FLOOR_HZ))
            hi = np.log2(np.maximum(f_hi, _LOG_FLOOR_HZ))
        b0 = np.clip(np.floor((lo - self.log_edges[0]) / step + 1e-9), 0, self.n_bins).astype(int)
        b1 = np.clip(np.ceil((hi - self.log_edges[0]) / step - 1e-9), 0, self.n_bins).astype(int)
        table = np.zeros((self.n_slots + 1, self.n_bins + 1), dtype=np.int64)
        table[1:, 1:] = np.cumsum(np.cumsum(self.cells, axis=0), axis=1)
        count = table[s1, b1] - table[s0, b1] - table[s1, b0] + table[s0, b0]
        out = np.zeros(n, bool)
        valid = (s1 > s0) & (b1 > b0)
        out[valid] = count[valid] > 0
        return out

    @property
    def area(self):
        """Occupied cells times a cell's area, s x octaves."""
        step = self.log_edges[1] - self.log_edges[0]
        return float(self.cells.sum()) * self.slot_s * step

    def record(self):
        """The mask as plain arrays, for an npz."""
        return dict(mask_t0=self.t0, mask_slot_s=self.slot_s, mask_log_edges=self.log_edges,
                    mask_cells=self.cells)


def _band_of(row):
    """First and last occupied cell of a mask row."""
    idx = np.flatnonzero(row)
    return int(idx[0]), int(idx[-1])


def tile_coverage(tiles, settings, rate):
    """The event's tile coverage: the fraction of its span's cells that its
    tiles occupy.

    The tiles are marked on the mask's grid (`CompletionSlot` by
    `CompletionOctaveStep`, each tile's whole support); the span is that
    mask with the holes between occupied time cells bridged
    (`CompletionBridgeGap`) and, in each time cell, the cells from the lowest
    occupied one to the highest. The number is read from the tiles alone --
    no dilation, no track, no stream -- so it is the same online and
    offline, and symmetric in time and in sweep direction. One minus it is
    the share of the span the completion fills.

    :param tiles: ``(t_lo, t_hi, f_lo, f_hi[, ...])`` of the event's tiles.
    :type settings: CompletionSettings
    :type rate: float
    :param rate: sampling rate of the stream, Hz.
    :return: float -- in [0, 1]; NaN without tiles.
    """
    t_lo, t_hi, f_lo, f_hi = (np.asarray(a, dtype=float).reshape(-1) for a in tiles[:4])
    if not t_lo.size:
        return float("nan")
    mask = CompletionMask.empty(float(t_lo.min()), float(t_hi.max()), max(settings.low_frequency, _LOG_FLOOR_HZ),
                                0.5 * float(rate), settings.slot_s, settings.octave_step)
    mask.add_boxes(t_lo, t_hi, f_lo, f_hi)
    tiles_cells = int(mask.cells.sum())
    if not tiles_cells:
        return float("nan")
    mask.bridge(settings.bridge_gap)
    return tiles_cells / float(mask.spanned().sum())


def build_mask(tiles, settings, rate, track=None, extent=None):
    """The completion mask of an event: its tiles, bridged, dilated, with the
    track's band.

    :param tiles: ``(t_lo, t_hi, f_lo, f_hi[, energy, ...])`` of the event's
        tiles, GPS and Hz (`detector_graph.event_tiles`); the energy is read
        by "centroid".
    :type settings: CompletionSettings
    :param settings: the widths.
    :type rate: float
    :param rate: sampling rate of the stream, Hz (the mask's ceiling).
    :param track: ``(times, frequencies)`` of a ridge to add, or None.
    :param extent: ``(start, end)`` GPS of the event, read from the tiles and
        the track when None.
    :return: CompletionMask
    :raises ValueError: without tiles and track.
    """
    t_lo, t_hi, f_lo, f_hi = (np.asarray(a, dtype=float).reshape(-1) for a in tiles[:4])
    times = [t_lo, t_hi]
    if track is not None:
        tt = np.asarray(track[0], dtype=float)
        times.append(tt[np.isfinite(tt)])
    all_t = np.concatenate(times) if times else np.zeros(0)
    if extent is None:
        if not all_t.size:
            raise ValueError("a completion needs tiles or a track")
        extent = (float(all_t.min()), float(all_t.max()))
    start = float(extent[0]) - settings.pad
    end = float(extent[1]) + settings.pad
    mask = CompletionMask.empty(start, end, max(settings.low_frequency, _LOG_FLOOR_HZ), 0.5 * float(rate),
                                settings.slot_s, settings.octave_step)
    if settings.tile_band == "centroid" and len(tiles) > 4:
        mask.add_centroids(t_lo, t_hi, f_lo, f_hi, np.asarray(tiles[4], dtype=float).reshape(-1),
                           settings.band)
    else:
        mask.add_boxes(t_lo, t_hi, f_lo, f_hi)
    mask.bridge(settings.bridge_gap)
    if track is not None:
        mask.add_ridge(track[0], track[1], settings.band)
    mask.dilate(settings.dilation_time, settings.dilation_octaves)
    return mask


# ================================================================ inversion
def _piece(stream, fs, start, end, margin):
    """A power-of-2 piece of the stream holding ``[start, end)`` with
    `margin` samples on each side, zero where the stream does not reach:
    (GPS of its first sample, samples, index of `start` in it)."""
    t0, x = stream
    x = np.asarray(x, dtype=float)
    need = int(np.ceil((end - start) * fs)) + 2 * int(margin)
    n = 1 << int(np.ceil(np.log2(max(need, 2))))
    centre = int(round((0.5 * (start + end) - t0) * fs))
    a = centre - n // 2
    piece = np.zeros(n)
    lo, hi = max(a, 0), min(a + n, x.size)
    if hi > lo:
        piece[lo - a:hi - a] = x[lo:hi]
    piece[~np.isfinite(piece)] = 0.0
    return t0 + a / fs, piece, int(round((start - t0) * fs)) - a


def _soft(c, level):
    """Soft threshold at `level`."""
    if level <= 0:
        return c
    return np.sign(c) * np.maximum(np.abs(c) - level, 0.0)


def _packets(piece, fs, mask, start_gps, settings):
    """Mask inversion in the cosine-packet best basis."""
    from py4tsa.tsa import CosinePackets

    n = piece.size
    m = packet_segment(settings.basis)
    cp = CosinePackets(n, m)
    fine = np.asarray(cp.Analyse(piece, [m] * (n // m)), dtype=float).reshape(n // m, m)
    above = (np.arange(m) + 1) * fs / (2.0 * m) > settings.low_frequency + 1e-9
    sigma = max(float(np.median(np.abs(fine[:, above])) * MAD_TO_SIGMA), 1e-300)
    cp.BestBasis(piece / sigma, getattr(CosinePackets.Cost, settings.packet_cost), 1.0)
    seg = [int(s) for s in cp.GetSegmentation()]
    c = np.array(cp.GetCoefficients(), dtype=float) * sigma
    t_lo = np.empty(n)
    t_hi = np.empty(n)
    f_lo = np.empty(n)
    f_hi = np.empty(n)
    pos = 0
    for length in seg:
        k = np.arange(length)
        t_lo[pos:pos + length] = start_gps + pos / fs
        t_hi[pos:pos + length] = start_gps + (pos + length) / fs
        f_lo[pos:pos + length] = k * fs / (2.0 * length)
        f_hi[pos:pos + length] = (k + 1) * fs / (2.0 * length)
        pos += length
    keep = mask.keeps(t_lo, t_hi, f_lo, f_hi)
    kept = np.where(keep, _soft(c, settings.threshold * sigma), 0.0)
    y = np.asarray(cp.Inverse(kept / sigma, seg), dtype=float) * sigma
    return y, sigma, int(keep.sum()), dict(segmentation=np.asarray(seg))


def _fixed_basis(piece, fs, mask, start_gps, settings):
    """Mask inversion in a local cosine or p4TSA wavelet basis, the piece
    transformed as one window."""
    from wdf.analysis.ridge_tracking import dense_coefficients, tile_geometry, window_sigma

    n = piece.size
    basis = settings.basis
    c = dense_coefficients(piece[None, :], basis, fs)[0]
    g = tile_geometry(n, fs, basis)
    sigma = max(float(window_sigma(c[None, :], g["f_hi"], settings.low_frequency)[0]), 1e-300)
    keep = mask.keeps(start_gps + g["t_lo"], start_gps + g["t_hi"], g["f_lo"], g["f_hi"])
    kept = np.where(keep, _soft(c, settings.threshold * sigma), 0.0)
    if basis_kind(basis) == "localcos":
        from wdf.analysis.local_cosine import inverse_local_cosine
        from wdf.analysis.wavelets import _local_cosine_segment
        y = inverse_local_cosine(kept, _local_cosine_segment(basis))
    else:
        from wdf.analysis.reconstruction import inverse_transform
        y = inverse_transform(kept, basis)
    return np.asarray(y, dtype=float), sigma, int(keep.sum()), {}


def masked_pursuit(samples, start, fs, mask, settings, sigma=None):
    """Matching pursuit of sine-Gaussian atoms inside a mask, on a whitened
    stream.

    The atoms are those of `glitch_subtraction.SineGaussianBank` with unit
    taps (the stream is already white), `settings.qs` by
    `settings.frequency_count` frequencies log-spaced over the mask's band. At
    each step the atom and centre of largest SNR among those whose (centre,
    frequency) lies in the mask is taken, until `settings.stop_snr` or
    `settings.max_atoms`.

    :param samples: the whitened piece.
    :param start: GPS time of its first sample.
    :param fs: sampling rate, Hz.
    :type mask: CompletionMask
    :param settings: CompletionSettings
    :param sigma: the piece's noise scale; its robust standard deviation when
        None.
    :return: tuple -- (model on the piece's samples, list of
        `glitch_subtraction.Atom`, sigma).
    """
    from scipy.signal import fftconvolve

    from wdf.processes.glitch_subtraction import Atom, SineGaussianBank

    x = np.asarray(samples, dtype=float)
    if sigma is None:
        sigma = max(float(np.median(np.abs(x - np.median(x))) * MAD_TO_SIGMA), 1e-300)
    top = min(mask.f_high, 0.5 * fs * (1 - 1e-9))
    low = max(mask.f_low, _LOG_FLOOR_HZ)
    freqs = np.geomspace(low, top, settings.frequency_count, endpoint=False)[1:] if settings.frequency_count > 1 \
        else np.array([np.sqrt(low * top)])
    bank = SineGaussianBank(np.array([1.0]), 1.0, fs, freqs, qs=settings.qs)
    n = x.size
    q = x / sigma                                   # the pursuit runs in noise units
    model = np.zeros(n)
    atoms = []
    centres = start + np.arange(n) / fs
    allowed = {k: mask.contains(centres, np.full(n, a["frequency"])) for k, a in enumerate(bank.atoms)}
    for _ in range(int(settings.max_atoms)):
        best = (0.0, None)
        for k, a in enumerate(bank.atoms):
            ok = allowed[k]
            if not ok.any():
                continue
            half = a["half"]
            c = fftconvolve(q, a["g"][::-1], mode="same")           # centred on each sample
            cr, ci = c.real, c.imag
            gi = a["gram_inv"]
            snr2 = gi[0, 0] * cr * cr + 2.0 * gi[0, 1] * cr * ci + gi[1, 1] * ci * ci
            snr2 = np.where(ok, snr2, -1.0)
            j = int(np.argmax(snr2))
            if snr2[j] > best[0] ** 2 and half < j < n - half - 1:
                best = (float(np.sqrt(max(snr2[j], 0.0))), (k, j, cr[j], ci[j]))
        if best[1] is None or best[0] < settings.stop_snr:
            break
        k, j, cr, ci = best[1]
        a = bank.atoms[k]
        alpha, beta = a["gram_inv"] @ np.array([cr, ci])
        half = a["half"]
        wave = alpha * a["g"].real + beta * a["g"].imag
        model[j - half:j + half + 1] += wave * sigma
        q[j - half:j + half + 1] -= wave
        atoms.append(Atom(centre=start + j / fs, frequency=a["frequency"], q=a["q"],
                          alpha=float(alpha * sigma), beta=float(beta * sigma), snr=best[0],
                          rate=fs, half_length=int(half)))
    return model, atoms, sigma


# ================================================================ result
@dataclass
class CompletedWaveform:
    """The completed waveform of one event in one detector.

    :param t0: GPS time of the first sample.
    :param fs: sampling rate, Hz.
    :param waveform: the completed whitened waveform, in the stream's units.
    :param mask: the mask inverted.
    :param basis: the basis used ("mp": the atoms' bank).
    :param method: "mask" or "mp".
    :param sigma: noise scale of the piece, stream units.
    :param n_kept: coefficients inside the mask (atoms for "mp").
    :param n_total: coefficients of the piece.
    :param energy: sum of the squared waveform over sigma squared.
    :param coverage: fraction of the reference ridge the mask holds, NaN
        without one.
    :param sparse_overlap: overlap with the sparse reconstruction, NaN
        without one.
    :param atoms: the atoms of "mp".
    :param extra: the best basis's segmentation, when packets.
    :param tile_coverage: the event's tile coverage (`tile_coverage`).
    :param completed: True when the waveform is the completion; False when
        the gate (`CompletionMaxTileCoverage`) returned the sparse
        reconstruction (`method` "sparse", `n_kept` 0).
    """

    t0: float
    fs: float
    waveform: np.ndarray
    mask: CompletionMask
    basis: str
    method: str
    sigma: float
    n_kept: int
    n_total: int
    energy: float
    coverage: float = float("nan")
    sparse_overlap: float = float("nan")
    atoms: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    tile_coverage: float = float("nan")
    completed: bool = True

    @property
    def t_end(self):
        return self.t0 + self.waveform.size / self.fs

    def times(self):
        """GPS time of each sample."""
        return self.t0 + np.arange(self.waveform.size) / self.fs

    def on_grid(self, t0, size):
        """The waveform laid on another grid at the same rate, zero outside."""
        return place(self.t0, self.waveform, t0, size, self.fs)

    def strain(self, taps, band, bandpass=None):
        """The strain whose zero-phase whitening gives the waveform
        (`hrss.strain_from_whitened`), on the waveform's own samples.

        :param taps: the symmetric whitening taps of the model in force
            (`hrss.ModelHistory.taps`); the waveform must be in the units the
            taps produce (not divided by sigma).
        :param band: ``(f_lo, f_hi)``, Hz.
        :param bandpass: `BandPassDownSampling` to divide out, or None.
        :return: numpy.ndarray -- strain, same length as the waveform.
        """
        from wdf.analysis.hrss import strain_from_whitened

        strain, _ = strain_from_whitened(self.waveform, self.fs, taps, band, bandpass)
        k = 2 * np.asarray(taps).size
        return np.asarray(strain, dtype=float)[k:k + self.waveform.size]

    def record(self):
        """Scalars of the result as a dict."""
        return dict(t0=self.t0, fs=self.fs, basis=self.basis, method=self.method, sigma=self.sigma,
                    n_kept=self.n_kept, n_total=self.n_total, energy=self.energy,
                    coverage=self.coverage, sparse_overlap=self.sparse_overlap,
                    n_samples=int(self.waveform.size), n_atoms=len(self.atoms),
                    tile_coverage=self.tile_coverage, completed=self.completed)


def energy_ridge(samples, t0, fs, m, f_low, floor):
    """A model-free ridge of a waveform: in each segment of its local cosine
    map (``LocalCos<m>``), the energy-weighted log-frequency centroid above
    `f_low`, for the segments holding at least `floor` of the loudest
    segment's energy.

    :return: tuple -- (GPS times of the segments' centres, frequencies, Hz).
    """
    from wdf.analysis.local_cosine import local_cosine

    x = np.asarray(samples, dtype=float)
    n = (x.size // m) * m
    if n < m:
        return np.zeros(0), np.zeros(0)
    c = local_cosine(x[:n], m).reshape(m, n // m)
    f = (np.arange(m) + 0.5) * fs / (2.0 * m)
    e = np.where(f[:, None] > f_low, c ** 2, 0.0)
    total = e.sum(axis=0)
    keep = total >= floor * total.max() if total.max() > 0 else np.zeros(n // m, bool)
    lf = (e * np.log2(f)[:, None]).sum(axis=0)[keep] / total[keep]
    times = t0 + (np.flatnonzero(keep) + 0.5) * m / fs
    return times, 2.0 ** lf


def hole_fraction(reconstruction, reference):
    """The fraction of `reference`'s energy lying where `reconstruction` is
    exactly zero: what the holes of a reconstruction leave out of a known
    waveform on the same grid."""
    a = np.asarray(reconstruction, dtype=float)
    b = np.asarray(reference, dtype=float) ** 2
    total = float(b.sum())
    return float(b[a == 0.0].sum() / total) if total > 0 else float("nan")


def place(t0, y, grid_t0, size, fs):
    """`y` (first sample at `t0`) on a grid of `size` samples from `grid_t0`
    at `fs`, zero outside."""
    out = np.zeros(int(size))
    y = np.asarray(y, dtype=float)
    i0 = int(round((t0 - grid_t0) * fs))
    a, b = max(i0, 0), min(i0 + y.size, int(size))
    if b > a:
        out[a:b] = y[a - i0:b - i0]
    return out


def complete(stream, fs, tiles, settings, track=None, extent=None, reference=None, sparse=None):
    """The completed waveform of an event from its tiles and the whitened
    stream.

    :param stream: ``(GPS of the first sample, samples)`` of the detector's
        zero-phase whitened stream, holding the event with
        `settings.pad` and the basis's margin on each side (the piece is zero
        where it does not reach).
    :type fs: float
    :param fs: sampling rate, Hz.
    :param tiles: ``(t_lo, t_hi, f_lo, f_hi[, ...])`` of the event's tiles
        (`detector_graph.event_tiles`).
    :type settings: CompletionSettings
    :param track: ``(times, frequencies)`` of the event's track, or None.
    :param extent: ``(start, end)`` GPS of the event; from the tiles and the
        track when None.
    :param reference: ``(times, frequencies)`` of the ridge the coverage is
        measured on (the track, or an injected one); `track` when None.
    :param sparse: ``(GPS of the first sample, samples)`` of the sparse
        reconstruction (`detector_graph.event_waveform`), for the overlap
        and for the gate.
    :return: CompletedWaveform -- the completion when the event's tile
        coverage is at or below `settings.max_tile_coverage` (or no `sparse`
        is given); otherwise `sparse` as it is, with ``completed=False``.
    """
    covered = tile_coverage(tiles, settings, fs)
    mask = build_mask(tiles, settings, fs, track=track, extent=extent)
    start, end = mask.t0, mask.t_end
    ridge = reference if reference is not None else track
    if sparse is not None and np.isfinite(covered) and covered > settings.max_tile_coverage:
        t0, y = float(sparse[0]), np.asarray(sparse[1], dtype=float).copy()
        y[~np.isfinite(y)] = 0.0
        x = _piece(stream, fs, start, end, 0)[1]
        sigma = max(float(np.median(np.abs(x - np.median(x))) * MAD_TO_SIGMA), 1e-300) if x.any() else float("nan")
        result = CompletedWaveform(t0=t0, fs=float(fs), waveform=y, mask=mask, basis="sparse", method="sparse",
                                   sigma=sigma, n_kept=0, n_total=0,
                                   energy=float(np.sum(y ** 2) / sigma ** 2) if np.isfinite(sigma) and sigma ** 2 > 0 else float("nan"),
                                   sparse_overlap=1.0, tile_coverage=covered, completed=False)
        if ridge is not None:
            result.coverage = mask.coverage(ridge[0], ridge[1])
        return result
    kind = basis_kind(settings.basis)
    if kind == "packets":
        margin = settings.margin_segments * packet_segment(settings.basis)
    elif kind == "localcos":
        from wdf.analysis.wavelets import _local_cosine_segment
        margin = settings.margin_segments * _local_cosine_segment(settings.basis)
    else:
        margin = int(round(settings.wavelet_margin * fs))
    piece_t0, piece, i0 = _piece(stream, fs, start, end, margin)
    size = int(np.ceil((end - start) * fs))
    atoms, extra = [], {}
    if settings.method == "mp":
        y, atoms, sigma = masked_pursuit(piece, piece_t0, fs, mask, settings)
        n_kept, n_total, basis = len(atoms), piece.size, "sine-Gaussian"
    elif kind == "packets":
        y, sigma, n_kept, extra = _packets(piece, fs, mask, piece_t0, settings)
        n_total, basis = piece.size, settings.basis
    else:
        y, sigma, n_kept, extra = _fixed_basis(piece, fs, mask, piece_t0, settings)
        n_total, basis = piece.size, settings.basis
    out = np.zeros(size)
    a, b = max(i0, 0), min(i0 + size, piece.size)
    if b > a:
        out[a - i0:b - i0] = y[a:b]
    out[~np.isfinite(out)] = 0.0
    result = CompletedWaveform(t0=start, fs=float(fs), waveform=out, mask=mask, basis=basis,
                               method=settings.method, sigma=float(sigma), n_kept=int(n_kept),
                               n_total=int(n_total), energy=float(np.sum(out ** 2) / sigma ** 2) if sigma ** 2 > 0 else float("nan"),
                               atoms=atoms, extra=extra, tile_coverage=covered)
    if ridge is not None:
        result.coverage = mask.coverage(ridge[0], ridge[1])
    if sparse is not None:
        from wdf.analysis.reconstruction import waveform_overlap
        s = place(sparse[0], sparse[1], start, size, fs)
        result.sparse_overlap = float(waveform_overlap(out, s)["overlap"])
    return result


def event_members(labels, cluster_id):
    """Row positions of the event `cluster_id` among `labels`."""
    return np.flatnonzero(np.asarray(labels) == int(cluster_id))


def complete_event(graph, labels, cluster_id, stream, fs, settings, track=None, reference=None,
                   with_sparse=True):
    """`complete` for one event of a detector graph: its tiles are the
    surviving coefficients of its members (`event_tiles`), its sparse
    reconstruction `event_waveform`.

    :param graph: `detector_graph.DetectorGraph` of the detector.
    :param labels: component label per node (`graph.components()`).
    :param cluster_id: the event.
    :param stream: ``(GPS of the first sample, samples)`` whitened.
    :param fs: sampling rate, Hz.
    :param settings: CompletionSettings
    :param track: ``(times, frequencies)`` of the event's track, or None.
    :param reference: ridge for the coverage, or None (the track).
    :param with_sparse: compute the sparse reconstruction and its overlap.
    :return: tuple -- (CompletedWaveform, sparse ``(t0, samples)`` or None).
    """
    from wdf.analysis.detector_graph import event_tiles, event_waveform

    members = event_members(labels, cluster_id)
    tiles = event_tiles(graph.nodes, members)
    sparse = None
    if with_sparse:
        got = event_waveform(graph, labels, cluster_id)
        if int(cluster_id) in got:
            sparse = got[int(cluster_id)][:2]
    result = complete(stream, fs, tiles, settings, track=track, reference=reference, sparse=sparse)
    return result, sparse


# ================================================================ the stream
def conditioned_stretch(strain, rate_in, start, bandpass, block_s):
    """The conditioned stream of a stretch of raw strain, as the online worker
    conditions it: `bandpass` (`BandPassDownSampling`, zero phase) fed one
    block at a time; the first block it emits, which has no real past, is
    dropped.

    :param strain: raw samples at `rate_in`.
    :param rate_in: their sampling rate, Hz.
    :param start: GPS time of the first sample; on the run's block grid when
        the worker's samples are wanted bit for bit.
    :param bandpass: a fresh `BandPassDownSampling` (it holds state).
    :param block_s: the worker's block (`len`), s.
    :return: tuple -- (samples, GPS of the first, rate).
    """
    from wdf.processes.BandPassDownSampling import SV_to_array
    from wdf.structures.array2SeqView import array2SeqView

    x = np.asarray(strain, dtype=float)
    n_blk = int(round(block_s * float(rate_in)))
    out, t0, first = [], None, True
    for k in range(0, x.size - n_blk + 1, n_blk):
        t = float(start) + k / float(rate_in)
        view = array2SeqView(t, float(rate_in), n_blk)
        view.Fill(t, x[k:k + n_blk].copy())
        y = bandpass.Process(view.SV)
        if y is None:
            continue
        if first:
            first = False
            continue
        t0 = float(y.GetStart()) if t0 is None else t0
        out.append(SV_to_array(y) * y.GetScale())
    if not out:
        return np.zeros(0), float("nan"), float(bandpass.resampling)
    return np.concatenate(out), t0, float(bandpass.resampling)


def whitened_stretch(strain, rate_in, start, history, bandpass=None, block_s=None, held=None,
                     normalise=False):
    """The zero-phase whitened stream of a stretch of strain, as the online
    worker whitens it: the conditioning (`conditioned_stretch`) when
    `bandpass` is given, then the symmetric taps of the model in force at each
    sample (`hrss.ModelHistory`), by convolution; the edges the taps cannot
    fill are removed.

    The same call on an injected strain gives the injection as the search
    saw it.

    :param strain: raw samples at `rate_in`, holding half the taps' length
        and two conditioning blocks beyond the stretch wanted on each side.
    :param rate_in: their sampling rate, Hz.
    :param start: GPS time of the first sample.
    :type history: hrss.ModelHistory
    :param history: the models the run applied.
    :param bandpass: `BandPassDownSampling` to condition with, or None when
        `strain` is already conditioned at the models' rate.
    :param block_s: the worker's block (`len`), s; required with
        `bandpass`.
    :param held: `hrss._held` (``band``, ``blend``) when `HoldOutsideBand`
        was on, else None.
    :param normalise: divide each sample by its model's sigma (the units the
        searches read, "stream over sigma").
    :return: tuple -- (GPS of the first output sample, whitened samples,
        rate).
    """
    from scipy.signal import fftconvolve

    if bandpass is not None:
        if block_s is None:
            raise ValueError("the conditioning needs the worker's block, block_s")
        x, t0, rate = conditioned_stretch(strain, rate_in, start, bandpass, block_s)
    else:
        x, t0, rate = np.asarray(strain, dtype=float), float(start), float(rate_in)
    if abs(rate - history.resampling) > 1e-9:
        raise ValueError(f"the stream is at {rate} Hz, the models at {history.resampling} Hz")
    k = int(np.asarray(history.taps(0, held)).size // 2)
    n = x.size - 2 * k
    if n <= 0:
        return t0 + k / rate, np.zeros(0), rate
    out = np.zeros(n)
    first = t0 + k / rate
    a = 0
    while a < n:
        i = history.index_at(first + a / rate)
        b = n
        if i + 1 < len(history.models):
            b = min(n, int(np.ceil((history.switch[i + 1] - first) * rate - 1e-9)))
        b = max(b, a + 1)
        taps = np.asarray(history.taps(i, held), dtype=float)
        y = fftconvolve(x[a:b + 2 * k], taps, mode="valid")
        if normalise:
            y = y / float(history.models[i]["sigma"])
        out[a:b] = y[:b - a]
        a = b
    return first, out, rate
