"""The wavelet coefficients of every trigger in a cluster, kept together.

A cluster is a set of consecutive analysis windows that the wavegram says
belong to the same transient. Reducing it to a handful of scalars -- a peak
time, a band, a statistic -- throws away the thing the search actually
measured: which coefficient carried what, in which window, in which basis.

`ClusterCoefficients` keeps the whole set as one array of shape
``(n_triggers, n_coeff)``. It is not symmetric and it is not square: the rows
are analysis windows, ordered in time and overlapping each other, and the
columns are coefficient indices, each of which is a tile in time and frequency
within its own window. Everything downstream of the search is a reduction of
this array -- the reconstructed waveform, the estimated parameters, and the
node features a learned coincidence model sees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wdf.analysis.coefficients import coefficient_matrix, from_dense, window_length


def _rows_by_label(frame: pd.DataFrame, cluster_column: str) -> dict:
    """Map each cluster label to the rows of `frame` that carry it.

    :type frame: pandas.DataFrame
    :param frame: triggers carrying a cluster label.
    :type cluster_column: str
    :param cluster_column: the label column.
    :return: dict -- ``{label: numpy.ndarray of row positions}``.
    """
    if cluster_column not in frame or len(frame) == 0:
        return {}
    label = frame[cluster_column].to_numpy(dtype=np.int64)
    order = np.argsort(label, kind="stable")
    ordered = label[order]
    boundary = np.flatnonzero(np.diff(ordered)) + 1
    return dict(zip((int(v) for v in ordered[np.concatenate(([0], boundary))]),
                    np.split(order, boundary)))


@dataclass
class ClusterCoefficients:
    """One cluster's wavelet coefficients, one row per analysis window.

    :param cluster_id: the cluster's label in its detector.
    :param ifo: detector name.
    :param fs: analysis sampling frequency, Hz.
    :param window: analysis window length, samples.
    :param overlap: overlap between consecutive windows, samples.
    :param times: GPS time of each window's first sample, ascending.
    :param coefficients: ``(n_triggers, n_coeff)`` array of coefficients, in
        the same row order as `times`.
    :param waves: the winning basis of each window.
    :param sigma: the noise scale of each window.
    """

    cluster_id: int
    ifo: str
    fs: float
    window: int
    overlap: int
    times: np.ndarray
    coefficients: np.ndarray
    waves: tuple[str, ...]
    sigma: np.ndarray

    @property
    def n_triggers(self) -> int:
        """Number of analysis windows in the cluster."""
        return int(self.coefficients.shape[0])

    @property
    def n_coeff(self) -> int:
        """Number of coefficients per window."""
        return int(self.coefficients.shape[1])

    @property
    def gps_start(self) -> float:
        """GPS time of the first sample the cluster covers."""
        return float(self.times[0])

    @property
    def gps_end(self) -> float:
        """GPS time of the last sample the cluster covers."""
        return float(self.times[-1] + self.window / self.fs)

    @property
    def duration(self) -> float:
        """Span of the cluster in seconds."""
        return self.gps_end - self.gps_start

    @property
    def noise_scale(self) -> float:
        """One noise scale for the cluster: the median over its windows."""
        finite = self.sigma[np.isfinite(self.sigma) & (self.sigma > 0.0)]
        return float(np.median(finite)) if finite.size else float("nan")

    @classmethod
    def from_triggers(cls, triggers: pd.DataFrame, fs: float, window: int, overlap: int,
                      cluster_id: int = -1, ifo: str = "") -> "ClusterCoefficients":
        """Assemble the array from the triggers of one cluster.

        :type triggers: pandas.DataFrame
        :param triggers: the cluster's member triggers, carrying `gps`, `wave`,
            `sigma` and the coefficient columns.
        :type fs: float
        :param fs: analysis sampling frequency, Hz.
        :type window: int
        :param window: analysis window length, samples.
        :type overlap: int
        :param overlap: overlap between consecutive windows, samples.
        :type cluster_id: int
        :param cluster_id: the label to record on the result.
        :type ifo: str
        :param ifo: detector name; taken from the triggers when they carry one.
        :return: ClusterCoefficients
        :raises ValueError: if `triggers` is empty.
        """
        if len(triggers) == 0:
            raise ValueError("a cluster needs at least one trigger")

        ordered = triggers.sort_values("gps")

        return cls(
            cluster_id=int(cluster_id),
            ifo=str(ordered["ifo"].iloc[0]) if "ifo" in ordered else ifo,
            fs=float(fs),
            window=int(window),
            overlap=int(overlap),
            times=ordered["gps"].to_numpy(dtype=float),
            # Single precision: the coefficients carry seven significant
            # digits, far more than anything read off them needs, and a full
            # segment holds hundreds of thousands of these rows. The
            # reconstruction converts back to double, where py4tsa wants it.
            coefficients=coefficient_matrix(ordered),
            waves=tuple(ordered["wave"].astype(str)) if "wave" in ordered else (),
            sigma=(ordered["sigma"].to_numpy(dtype=float) if "sigma" in ordered
                   else np.full(len(ordered), np.nan)),
        )

    def reconstruct(self) -> tuple[float, np.ndarray]:
        """Invert the coefficients and stitch the windows into one time series.

        The series is a function of the coefficients, which are fixed when the
        cluster is assembled, so it is inverted once and kept. The statistic
        over the cluster is the norm of this series and the network stage times
        a pair on it: asking for both would otherwise invert every window of
        every event twice. Each caller receives its own samples.

        :return: ``(gps_start, samples)`` -- see `wdf.analysis.reconstruction`.
        """
        from wdf.analysis.reconstruction import stitch

        held = getattr(self, "_reconstruction", None)
        if held is None:
            held = stitch(self.as_frame(), self.fs, self.window, self.overlap)
            self._reconstruction = held
        start, samples = held
        return start, samples.copy()

    def enwdf(self) -> float:
        """The statistic over the whole cluster: the norm of the stitched
        reconstruction on the noise scale.

        Summing the coefficient energy across rows instead would double count,
        because consecutive windows overlap and a sample covered by two of them
        contributes to both.

        :return: float -- the cluster's `EnWDF`.
        """
        sigma = self.noise_scale
        if not np.isfinite(sigma) or sigma <= 0.0:
            return float("nan")
        _, samples = self.reconstruct()
        return float(np.linalg.norm(samples) / sigma)

    def wavegram(self, n_time_bins: int = 64) -> np.ndarray:
        """The cluster's tiles rendered on a fixed octave-by-time grid.

        Each coefficient is placed at its own tile's absolute time and octave,
        and the largest magnitude falling in a cell wins it. The grid has the
        same shape whatever the cluster's duration or number of windows, which
        is what lets a learned model take it as an input.

        :type n_time_bins: int
        :param n_time_bins: columns of the grid; rows are the transform's
            octave levels.
        :return: numpy.ndarray -- ``(n_levels, n_time_bins)`` of
            ``|coefficient| / sigma``, zero where no tile falls.
        """
        from wdf.analysis.wavelets import coeff_levels, coeff_time_bounds

        level, _ = coeff_levels(self.n_coeff)
        t_lo, t_hi = coeff_time_bounds(self.n_coeff, self.fs)
        t_mid = 0.5 * (t_lo + t_hi)

        n_levels = int(level.max()) + 2
        grid = np.zeros((n_levels, int(n_time_bins)))

        span = max(self.duration, 1.0 / self.fs)
        sigma = self.noise_scale
        if not (np.isfinite(sigma) and sigma > 0.0):
            # Without a scale the grid would be strain, of order 1e-22, which
            # every comparison downstream reads as zero. An empty grid says so.
            return grid
        scale = sigma

        row_of_level = np.where(level < 0, 0, level + 1)
        for row, start in enumerate(self.times):
            offset = float(start - self.gps_start)
            column = np.clip(((offset + t_mid) / span * n_time_bins).astype(int),
                             0, int(n_time_bins) - 1)
            magnitude = np.abs(self.coefficients[row]) / scale
            np.maximum.at(grid, (row_of_level, column), magnitude)

        return grid

    def as_frame(self) -> pd.DataFrame:
        """The cluster back as a trigger frame, for functions that take one.

        :return: pandas.DataFrame -- one row per window, with `gps`, `wave`,
            `sigma` and the coefficient columns.
        """
        pairs = [from_dense(row) for row in self.coefficients]
        return pd.DataFrame(dict(
            gps=self.times,
            wave=list(self.waves) if self.waves else "",
            sigma=self.sigma,
            ifo=self.ifo,
            n_coeff=self.n_coeff,
            fs=self.fs,
            wt_index=[index for index, _ in pairs],
            wt_value=[value for _, value in pairs],
        ))


def iter_cluster_coefficients(labeled_triggers: pd.DataFrame, events: pd.DataFrame,
                              fs: float, window: int, overlap: int,
                              cluster_column: str = "cluster_id"):
    """Yield ``(cluster_id, ClusterCoefficients)`` one event at a time.

    A segment holds hundreds of thousands of events, each carrying a matrix of
    its windows' coefficients, so building them all before deciding which ones
    matter costs gigabytes for nothing. Anything that only needs one at a time
    -- the statistic measured on the reconstruction, for instance -- should
    consume this rather than `collect_cluster_coefficients`.

    Arguments are the same as `collect_cluster_coefficients`.

    :return: iterator of ``(int, ClusterCoefficients)``.
    """
    if events.empty or labeled_triggers.empty:
        return

    ifo = str(labeled_triggers["ifo"].iloc[0]) if "ifo" in labeled_triggers else ""

    # The whole coefficient matrix is built once. Expanding one cluster's rows
    # out of the trigger frame instead costs an allocation per event, and a
    # segment holds hundreds of thousands of them.
    coefficients = coefficient_matrix(labeled_triggers)
    gps = labeled_triggers["gps"].to_numpy(dtype=float)
    waves = (labeled_triggers["wave"].astype(str).to_numpy()
             if "wave" in labeled_triggers else None)
    sigma = (labeled_triggers["sigma"].to_numpy(dtype=float)
             if "sigma" in labeled_triggers else np.full(len(gps), np.nan))

    labels = events[cluster_column].to_numpy(dtype=int)
    members_of = (events["member_indices"].to_numpy()
                  if "member_indices" in events else None)
    rows_of_label = _rows_by_label(labeled_triggers, cluster_column)

    # `member_indices` records each trigger's position in the frame the search
    # wrote, which is not its row in the cleaned and time-ordered frame.
    trigger_index = (labeled_triggers["trigger_index"].to_numpy(dtype=np.int64)
                     if "trigger_index" in labeled_triggers else None)
    if trigger_index is not None:
        row_of_trigger = np.full(int(trigger_index.max()) + 1, -1, dtype=np.int64)
        row_of_trigger[trigger_index] = np.arange(len(trigger_index))

    for position, label in enumerate(labels):
        members = None if members_of is None else members_of[position]
        if members is not None and len(members) and trigger_index is not None:
            rows = row_of_trigger[np.asarray(members, dtype=np.int64)]
            rows = rows[rows >= 0]
        else:
            rows = rows_of_label.get(int(label))
        if rows is None or len(rows) == 0:
            continue
        rows = rows[np.argsort(gps[rows], kind="stable")]
        yield int(label), ClusterCoefficients(
            cluster_id=int(label),
            ifo=ifo,
            fs=float(fs),
            window=int(window),
            overlap=int(overlap),
            times=gps[rows],
            coefficients=coefficients[rows],
            waves=tuple(waves[rows]) if waves is not None else (),
            sigma=sigma[rows],
        )


def iter_tile_cluster_coefficients(pixels: pd.DataFrame, labels, triggers: pd.DataFrame,
                                   window: int, overlap: int):
    """Yield ``(cluster_id, ClusterCoefficients)`` for events made of tiles.

    An event of `wdf.analysis.pixel_graph` owns tiles, not windows: one window
    can hold tiles of two events, and tiles the grouping joined to nothing.
    Its reconstruction inverts the event's own tiles and only those, so each
    window the event touches contributes a coefficient vector that is zero
    everywhere but at the tiles the event owns. The rest of the window's
    energy belongs to other events, or to none, and entering it would make the
    waveform, and the statistic read off it, describe something the grouping
    did not assemble.

    A region two overlapping windows both kept is one tile of the event but two
    estimates of it, and both are given here, one per window: the stitching
    averages them over the samples the windows share, which is what makes the
    norm of the series count the region once.

    The result is the same object the trigger-level grouping produces, so its
    statistic, its waveform and its map are computed by the same code.

    :type pixels: pandas.DataFrame
    :param pixels: the tile cloud, as `wdf.analysis.scale.pixel_cloud`
        returns it, carrying `trigger_index`, `coefficient` and `value`, at one
        window length.
    :param labels: the event of every row of `pixels`, positionally, as
        `wdf.analysis.pixel_graph.tile_labels` returns it; -1 for none.
    :type triggers: pandas.DataFrame
    :param triggers: the triggers the cloud was built from, indexed as they
        were when it was built (`trigger_index` is their index), carrying
        `gps`, `wave`, `sigma`, `n_coeff` and `fs`.
    :type window: int
    :param window: analysis window length, samples.
    :type overlap: int
    :param overlap: overlap between consecutive windows, samples.
    :return: iterator of ``(int, ClusterCoefficients)``, in increasing label.
    :raises KeyError: if a tile names a trigger `triggers` does not hold.
    :raises ValueError: if the cloud holds more than one window length, which
        no single coefficient vector can hold.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if pixels.empty or not (labels >= 0).any():
        return
    lengths = np.unique(pixels["scale"].to_numpy(dtype=float))
    if len(lengths) != 1:
        raise ValueError(
            f"the cloud holds {len(lengths)} window lengths; a coefficient "
            "vector belongs to one of them")
    n_coeff = int(lengths[0])

    keep = labels >= 0
    label = labels[keep]
    position = triggers.index.get_indexer(
        pixels["trigger_index"].to_numpy()[keep])
    if (position < 0).any():
        raise KeyError("a tile names a trigger the trigger table does not hold")
    coefficient = pixels["coefficient"].to_numpy(dtype=np.int64)[keep]
    value = pixels["value"].to_numpy(dtype=float)[keep]

    gps = triggers["gps"].to_numpy(dtype=float)
    waves = triggers["wave"].astype(str).to_numpy()
    sigma = triggers["sigma"].to_numpy(dtype=float)
    fs = float(triggers["fs"].to_numpy(dtype=float)[position[0]])
    ifo = str(pixels["ifo"].iloc[0]) if "ifo" in pixels else ""

    # Grouped by one sort over the labels rather than by a pass over the
    # cloud per event, which is quadratic in the event count.
    order = np.argsort(label, kind="stable")
    boundary = np.flatnonzero(np.diff(label[order])) + 1
    for rows in np.split(order, boundary):
        members, row_of = np.unique(position[rows], return_inverse=True)
        in_time = np.argsort(gps[members], kind="stable")
        rank = np.empty_like(in_time)
        rank[in_time] = np.arange(len(in_time))
        matrix = np.zeros((len(members), n_coeff), dtype=np.float32)
        matrix[rank[row_of], coefficient[rows]] = value[rows]
        members = members[in_time]
        cluster = int(label[rows[0]])
        yield cluster, ClusterCoefficients(
            cluster_id=cluster, ifo=ifo, fs=fs, window=int(window),
            overlap=int(overlap), times=gps[members], coefficients=matrix,
            waves=tuple(waves[members]), sigma=sigma[members])


def collect_cluster_coefficients(labeled_triggers: pd.DataFrame, events: pd.DataFrame,
                                 fs: float, window: int, overlap: int,
                                 cluster_column: str = "cluster_id") -> dict:
    """Build a `ClusterCoefficients` for every event in a detector's catalogue.

    Singletons are included: a transient short enough to fall inside one
    analysis window is a one-row array, not a missing one.

    :type labeled_triggers: pandas.DataFrame
    :param labeled_triggers: the triggers, carrying a cluster label and the
        ``wt*`` columns.
    :type events: pandas.DataFrame
    :param events: the event catalogue, used for its `cluster_id` and, when
        present, its `member_indices`.
    :type fs: float
    :param fs: analysis sampling frequency, Hz.
    :type window: int
    :param window: analysis window length, samples.
    :type overlap: int
    :param overlap: overlap between consecutive windows, samples.
    :type cluster_column: str
    :param cluster_column: the label column in `labeled_triggers`.
    :return: dict -- ``{cluster_id: ClusterCoefficients}``.
    """
    return dict(iter_cluster_coefficients(labeled_triggers, events, fs, window,
                                          overlap, cluster_column))


def score_events_by_reconstruction(events: pd.DataFrame, coefficients,
                                   cluster_column: str = "cluster_id") -> pd.DataFrame:
    """Re-score a detector's events on their stitched reconstruction.

    A fixed-window search scores each window on its own, so a transient longer
    than the window is divided and every piece is judged separately. Thresholding
    on that is thresholding on a fraction of the signal, and it is why long weak
    signals are the ones a raised threshold costs most: a binary neutron star can
    fail the cut in every single window while its whole reconstruction would pass
    it comfortably.

    Reconstructing across the windows an event spans and taking the norm on the
    noise scale gives the statistic over the signal's full extent, which is the
    quantity to threshold on. The per-window value is kept alongside it, so the
    gain is visible rather than asserted.

    :type events: pandas.DataFrame
    :param events: one detector's event catalogue, carrying `cluster_id`.
    :type coefficients: dict or iterable
    :param coefficients: either ``{cluster_id: ClusterCoefficients}`` as
        `collect_cluster_coefficients` returns, or the ``(cluster_id, cluster)``
        stream `iter_cluster_coefficients` yields -- the stream holds one event
        at a time, which is what a whole segment needs.
    :type cluster_column: str
    :param cluster_column: the label column in `events`.
    :return: pandas.DataFrame -- `events` with `EnWDF` now measured on the
        reconstruction, the per-window value preserved as `EnWDF_window`, and
        `n_windows` recording how many windows contributed.
    """
    if events.empty:
        return events.copy()

    out = events.reset_index(drop=True).copy()
    # The per-window value is what the grouping is judged against, so it is
    # kept as the catalogue reported it. Overwriting it with `EnWDF` would
    # replace it with the grouped estimate this function is about to improve.
    if "EnWDF_window" not in out:
        out["EnWDF_window"] = out["EnWDF"] if "EnWDF" in out else np.nan

    reconstructed = np.full(len(out), np.nan)
    n_windows = np.zeros(len(out), dtype=int)
    row_of_label = {label: row for row, label
                    in enumerate(out[cluster_column].to_numpy(dtype=int))}

    stream = coefficients.items() if hasattr(coefficients, "items") else coefficients
    for label, cluster in stream:
        row = row_of_label.get(int(label))
        if row is None:
            continue
        reconstructed[row] = cluster.enwdf()
        n_windows[row] = cluster.n_triggers

    # An event whose noise scale was not recorded keeps its per-window value
    # rather than becoming a hole in the catalogue.
    out["EnWDF"] = np.where(np.isfinite(reconstructed), reconstructed, out["EnWDF_window"])
    out["n_windows"] = n_windows
    return out
