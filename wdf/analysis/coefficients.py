"""The trigger record: the wavelet coefficients that survived thresholding.

WDF reports, for each analysis window, the statistic `EnWDF`, the basis that
won, the local noise scale, and the coefficients left standing after the
Donoho-Johnstone threshold. Almost all of them are zero -- the thresholding is
what the search does -- so the record is the `(index, value)` pairs of the
survivors, and everything else about the event is derived from them.

This module is the only place that knows how that record is laid out on disk.
Consumers ask for a coefficient vector or a coefficient matrix and are given
one; they do not discover columns, and they do not learn the storage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# The coefficient index is stored as an unsigned 16-bit integer, so a window
# may hold up to this many coefficients.
MAX_COEFF = 1 << 16

# A GPS time is a large number that has to stay meaningful to well under a
# millisecond, so it needs double precision. Everything else here is a
# measurement carrying a handful of significant digits, and single precision
# holds seven of them.
TIME_FIELDS = ["gps", "gpsStart", "gpsCentroid", "gpsPeak"]

MEASUREMENT_FIELDS = [
    "tSpread", "duration", "duration90", "EnWDF", "sigma", "snrPeak",
    "freqMin", "freqMean", "freqMax", "freqQ05", "freqQ95",
]

META_FIELDS = TIME_FIELDS + MEASUREMENT_FIELDS

COEFFICIENT_FIELDS = ["wt_index", "wt_value"]

TRIGGER_SCHEMA = pa.schema(
    [pa.field(name, pa.float64()) for name in TIME_FIELDS]
    + [pa.field(name, pa.float32()) for name in MEASUREMENT_FIELDS]
    + [
        pa.field("wave", pa.string()),
        pa.field("n_coeff", pa.uint32()),
        pa.field("fs", pa.float64()),
        pa.field("wt_index", pa.list_(pa.uint16())),
        pa.field("wt_value", pa.list_(pa.float32())),
    ]
)

# The file is mostly columns of measurements, whose mantissas do not compress.
# byte_stream_split regroups their bytes so that exponents sit together and
# zstd has something to work with; the basis name keeps dictionary encoding,
# which is what that is good at. pyarrow silently ignores the encoding unless
# dictionary encoding is turned off for the columns it applies to.
_FLOAT_COLUMNS = TIME_FIELDS + MEASUREMENT_FIELDS + ["fs"]

_WRITER_PROPERTIES = dict(
    compression="zstd",
    use_dictionary=["wave"],
    column_encoding={name: "BYTE_STREAM_SPLIT" for name in _FLOAT_COLUMNS},
)


def to_dense(index, value, n_coeff: int) -> np.ndarray:
    """Expand stored coefficient pairs into the full-length vector.

    :type index: array-like
    :param index: coefficient indices of the survivors.
    :type value: array-like
    :param value: coefficient values, in the same order as `index`.
    :type n_coeff: int
    :param n_coeff: length of the analysis window's coefficient vector.
    :return: numpy.ndarray -- length `n_coeff`, zero away from the survivors,
        with the dtype of `value`.
    :raises ValueError: if an index falls outside the vector.
    """
    index = np.asarray(index, dtype=np.int64).reshape(-1)
    value = np.asarray(value).reshape(-1)
    if len(index) != len(value):
        raise ValueError(f"{len(index)} indices for {len(value)} values")
    if len(index) and (index.min() < 0 or index.max() >= int(n_coeff)):
        raise ValueError(f"coefficient index outside a window of {n_coeff}")

    dense = np.zeros(int(n_coeff), dtype=value.dtype if value.size else np.float32)
    dense[index] = value
    return dense


def from_dense(coefficients) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a full-length coefficient vector to the pairs that survived.

    :type coefficients: array-like
    :param coefficients: one window's coefficients, zero where thresholded away.
    :return: tuple -- `(index, value)`, ascending in index.
    """
    coefficients = np.asarray(coefficients).reshape(-1)
    index = np.flatnonzero(coefficients)
    return index.astype(np.uint16), coefficients[index].astype(np.float32)


@dataclass
class SparseCoefficients:
    """One analysis window's surviving wavelet coefficients.

    :param index: coefficient indices of the survivors, ascending.
    :param value: coefficient values, in the same order as `index`.
    :param n_coeff: length of the window's coefficient vector.
    :param wave: name of the basis that produced them.
    :param sigma: the noise scale the search measured on this window.
    :param fs: sampling frequency the coefficients were computed at, Hz.
    """

    index: np.ndarray
    value: np.ndarray
    n_coeff: int
    wave: str
    sigma: float
    fs: float

    @classmethod
    def from_window(cls, coefficients, n_coeff: int, wave: str, sigma: float,
                    fs: float) -> "SparseCoefficients":
        """Build the record from a full-length coefficient vector.

        :type coefficients: array-like
        :param coefficients: the window's coefficients, zero where thresholded.
        :type n_coeff: int
        :param n_coeff: length of the coefficient vector.
        :type wave: str
        :param wave: name of the basis that produced them.
        :type sigma: float
        :param sigma: the noise scale the search measured on this window.
        :type fs: float
        :param fs: sampling frequency, Hz.
        :return: SparseCoefficients
        """
        index, value = from_dense(coefficients)
        return cls(index=index, value=value, n_coeff=int(n_coeff), wave=str(wave),
                   sigma=float(sigma), fs=float(fs))

    @property
    def n_nonzero(self) -> int:
        """Number of coefficients that survived thresholding."""
        return int(len(self.index))

    def dense(self) -> np.ndarray:
        """The full-length coefficient vector, zero away from the survivors."""
        return to_dense(self.index, self.value, self.n_coeff)

    def tiles(self) -> dict:
        """Time-frequency support and magnitude of each surviving coefficient.

        Times are relative to the start of the analysis window.

        :return: dict -- `t_lo`, `t_hi`, `f_lo`, `f_hi`, `magnitude`, each an
            array with one entry per survivor.
        """
        from wdf.analysis.wavelets import coeff_freq_bands, coeff_time_bounds

        t_lo, t_hi = coeff_time_bounds(self.n_coeff, self.fs)
        f_lo, f_hi = coeff_freq_bands(self.n_coeff, self.fs)
        index = np.asarray(self.index, dtype=np.int64)
        return dict(
            t_lo=t_lo[index],
            t_hi=t_hi[index],
            f_lo=f_lo[index],
            f_hi=f_hi[index],
            magnitude=np.abs(np.asarray(self.value, dtype=float)),
        )


class TriggerWriter:
    """Streams trigger records to a Parquet file, one row per trigger.

    Rows are buffered and written a row group at a time, so a segment's output
    never has to be held whole. `close()` finalises the file, which a Parquet
    reader needs before it can open it at all.
    """

    def __init__(self, path: str, flush_every: int = 20000):
        """
        :type path: str
        :param path: file to write; an existing file is replaced.
        :type flush_every: int
        :param flush_every: rows buffered before a row group is written. A row
            group is the unit the encodings work over, so small ones cost size;
            this is what bounds the writer's memory against that.
        """
        self.path = path
        self.flush_every = int(flush_every)
        self._rows = []
        self._writer = None

    def append(self, record: dict) -> None:
        """Buffer one trigger, writing a row group once enough have arrived.

        :type record: dict
        :param record: one row, carrying the fields of `TRIGGER_SCHEMA`.
        :raises ValueError: if the window is too long for the stored index type
            or an index falls outside it.
        """
        n_coeff = int(record["n_coeff"])
        if n_coeff > MAX_COEFF:
            raise ValueError(
                f"window of {n_coeff} coefficients exceeds the {MAX_COEFF} the "
                "stored index type can address")
        index = np.asarray(record["wt_index"], dtype=np.int64)
        if len(index) and (index.min() < 0 or index.max() >= n_coeff):
            raise ValueError(f"coefficient index outside a window of {n_coeff}")

        self._rows.append(record)
        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write the buffered rows as one row group."""
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=TRIGGER_SCHEMA)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, TRIGGER_SCHEMA,
                                            **_WRITER_PROPERTIES)
        self._writer.write_table(table)
        self._rows = []

    def close(self) -> None:
        """Write what is buffered and finalise the file."""
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def read_triggers(path: str) -> pd.DataFrame:
    """Read one trigger file.

    :type path: str
    :param path: Parquet file written by `TriggerWriter`.
    :return: pandas.DataFrame -- one row per trigger, with `wt_index` and
        `wt_value` as arrays.
    :raises ValueError: if the file does not carry the coefficient columns.
    """
    frame = pq.read_table(path).to_pandas()
    missing = [name for name in COEFFICIENT_FIELDS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} carries no coefficient columns {missing}")
    return frame


def window_length(frame: pd.DataFrame) -> int:
    """The coefficient-vector length the triggers share.

    :type frame: pandas.DataFrame
    :param frame: triggers carrying `n_coeff`.
    :return: int -- the common window length.
    :raises ValueError: if the triggers do not share one.
    """
    lengths = np.unique(np.asarray(frame["n_coeff"], dtype=np.int64))
    if len(lengths) != 1:
        raise ValueError(
            "triggers span more than one window length "
            f"({sorted(int(v) for v in lengths)}); the same coefficient index "
            "is a different tile at each of them, so they cannot share a matrix")
    return int(lengths[0])


def trigger_statistics(frame: pd.DataFrame) -> dict:
    """What a run produced, in the numbers worth reading before anything else.

    :type frame: pandas.DataFrame
    :param frame: one detector's triggers.
    :return: dict -- trigger count, livetime and rate; the quantiles of the
        surviving-coefficient count, of `EnWDF` and of `snrPeak`; how often each
        basis won; and the window lengths and sampling rates present, since a
        run may hold more than one of each.
    """
    if not len(frame):
        return dict(n_triggers=0)

    n_nonzero = np.fromiter((len(v) for v in frame["wt_index"]),
                            dtype=np.int64, count=len(frame))
    gps = pd.to_numeric(frame["gps"], errors="coerce").to_numpy()
    livetime = float(np.nanmax(gps) - np.nanmin(gps))

    quantiles = [0.5, 0.9, 0.99, 1.0]
    return dict(
        n_triggers=int(len(frame)),
        livetime_s=livetime,
        rate_hz=len(frame) / livetime if livetime > 0 else float("nan"),
        n_nonzero_mean=float(n_nonzero.mean()),
        n_nonzero_quantiles=dict(zip(
            quantiles, np.quantile(n_nonzero, quantiles).tolist())),
        density=float(n_nonzero.mean() / float(np.mean(frame["n_coeff"]))),
        enwdf_quantiles=dict(zip(
            quantiles, np.nanquantile(
                pd.to_numeric(frame["EnWDF"], errors="coerce"), quantiles).tolist())),
        snr_peak_quantiles=dict(zip(
            quantiles, np.nanquantile(
                pd.to_numeric(frame["snrPeak"], errors="coerce"), quantiles).tolist())),
        wave_counts=frame["wave"].value_counts().to_dict(),
        n_coeff=sorted(int(v) for v in np.unique(frame["n_coeff"])),
        fs=sorted(float(v) for v in np.unique(frame["fs"])),
    )


def coefficient_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The triggers' coefficients as one dense `(n_triggers, n_coeff)` array.

    :type frame: pandas.DataFrame
    :param frame: triggers carrying `n_coeff`, `wt_index` and `wt_value`.
    :return: numpy.ndarray -- single-precision coefficients, one row per
        trigger, zero away from the survivors.
    :raises ValueError: if the triggers do not share one window length.
    """
    n_coeff = window_length(frame)
    matrix = np.zeros((len(frame), n_coeff), dtype=np.float32)
    if not len(frame):
        return matrix

    index = frame["wt_index"].to_numpy()
    value = frame["wt_value"].to_numpy()
    counts = np.fromiter((len(v) for v in index), dtype=np.int64, count=len(index))
    if counts.sum():
        row = np.repeat(np.arange(len(frame)), counts)
        column = np.concatenate([np.asarray(v, dtype=np.int64) for v in index])
        matrix[row, column] = np.concatenate(
            [np.asarray(v, dtype=np.float32) for v in value])
    return matrix
