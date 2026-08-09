import numpy as np
import pytest

from wdf.analysis.coefficients import (
    MAX_COEFF,
    SparseCoefficients,
    TriggerWriter,
    coefficient_matrix,
    from_dense,
    read_triggers,
    to_dense,
    window_length,
)
from wdf.analysis.metaparameters import meta_features

# The window length is never a constant of the code, so everything that can be
# checked at more than one of them is.
WINDOWS = [128, 512]
FS = 2048.0


def sparse_window(n_coeff, rng, n_nonzero=4):
    index = np.sort(rng.choice(n_coeff, size=n_nonzero, replace=False))
    value = rng.normal(size=n_nonzero).astype(np.float32)
    return index.astype(np.uint16), value


def trigger_record(gps, index, value, n_coeff, sigma=1.0, wave="Daub4"):
    features = meta_features(index, value, n_coeff, FS, sigma, gps=gps)
    return dict(features, gps=gps, EnWDF=float(np.linalg.norm(value) / sigma),
                sigma=sigma, wave=wave, n_coeff=n_coeff, fs=FS,
                wt_index=[int(i) for i in index],
                wt_value=[float(v) for v in value])


@pytest.mark.parametrize("n_coeff", WINDOWS)
def test_dense_and_sparse_are_the_same_vector(n_coeff):
    rng = np.random.default_rng(0)
    dense = np.zeros(n_coeff, dtype=np.float32)
    index, value = sparse_window(n_coeff, rng)
    dense[index] = value

    recovered_index, recovered_value = from_dense(dense)
    assert np.array_equal(recovered_index, index)
    assert np.array_equal(recovered_value, value)
    assert np.array_equal(to_dense(recovered_index, recovered_value, n_coeff), dense)


@pytest.mark.parametrize("n_coeff", WINDOWS)
def test_the_round_trip_through_disk_is_exact(tmp_path, n_coeff):
    rng = np.random.default_rng(1)
    records = [trigger_record(1e9 + k, *sparse_window(n_coeff, rng), n_coeff)
               for k in range(7)]

    path = str(tmp_path / "triggers.parquet")
    # Fewer rows per group than records, so more than one row group is written.
    with TriggerWriter(path, flush_every=3) as writer:
        for record in records:
            writer.append(record)

    frame = read_triggers(path)
    assert len(frame) == len(records)
    for row, record in zip(frame.itertuples(), records):
        assert np.array_equal(np.asarray(row.wt_index), record["wt_index"])
        assert np.array_equal(np.asarray(row.wt_value, dtype=np.float32),
                              np.asarray(record["wt_value"], dtype=np.float32))
        assert row.gps == record["gps"]


@pytest.mark.parametrize("n_coeff", WINDOWS)
def test_the_matrix_is_the_dense_windows_stacked(tmp_path, n_coeff):
    rng = np.random.default_rng(2)
    windows = [sparse_window(n_coeff, rng) for _ in range(5)]
    records = [trigger_record(1e9 + k, index, value, n_coeff)
               for k, (index, value) in enumerate(windows)]

    path = str(tmp_path / "triggers.parquet")
    with TriggerWriter(path) as writer:
        for record in records:
            writer.append(record)

    matrix = coefficient_matrix(read_triggers(path))
    expected = np.stack([to_dense(index, value, n_coeff) for index, value in windows])
    assert matrix.shape == (len(windows), n_coeff)
    assert np.array_equal(matrix, expected)


def test_a_matrix_needs_one_window_length(tmp_path):
    rng = np.random.default_rng(3)
    path = str(tmp_path / "triggers.parquet")
    with TriggerWriter(path) as writer:
        for n_coeff in WINDOWS:
            writer.append(trigger_record(1e9, *sparse_window(n_coeff, rng), n_coeff))

    frame = read_triggers(path)
    with pytest.raises(ValueError, match="more than one window length"):
        window_length(frame)
    with pytest.raises(ValueError, match="more than one window length"):
        coefficient_matrix(frame)


def test_an_index_outside_the_window_is_refused(tmp_path):
    path = str(tmp_path / "triggers.parquet")
    record = trigger_record(1e9, np.array([3]), np.array([1.0], dtype=np.float32), 128)
    record["wt_index"] = [128]
    with TriggerWriter(path) as writer:
        with pytest.raises(ValueError, match="outside a window"):
            writer.append(record)


def test_a_window_too_long_for_the_index_type_is_refused(tmp_path):
    path = str(tmp_path / "triggers.parquet")
    record = trigger_record(1e9, np.array([3]), np.array([1.0], dtype=np.float32), 128)
    record["n_coeff"] = MAX_COEFF * 2
    with TriggerWriter(path) as writer:
        with pytest.raises(ValueError, match="exceeds"):
            writer.append(record)


@pytest.mark.parametrize("n_coeff", WINDOWS)
def test_the_record_reproduces_its_own_window(n_coeff):
    rng = np.random.default_rng(4)
    index, value = sparse_window(n_coeff, rng)
    record = SparseCoefficients(index=index, value=value, n_coeff=n_coeff,
                                wave="Daub4", sigma=1.0, fs=FS)
    assert record.n_nonzero == len(index)
    assert np.array_equal(record.dense(), to_dense(index, value, n_coeff))

    tiles = record.tiles()
    assert all(len(tiles[key]) == len(index) for key in tiles)
    assert np.all(tiles["t_hi"] > tiles["t_lo"])
    assert np.all(tiles["f_hi"] > tiles["f_lo"])
    assert np.all(tiles["f_hi"] <= FS / 2.0)


def test_the_measurements_are_single_precision_and_the_times_are_not(tmp_path):
    """A GPS time has to stay meaningful well under a millisecond; EnWDF, the
    noise scale and the frequencies carry a handful of significant digits."""
    rng = np.random.default_rng(5)
    path = str(tmp_path / "triggers.parquet")
    with TriggerWriter(path) as writer:
        writer.append(trigger_record(1.4e9, *sparse_window(512, rng), 512))

    frame = read_triggers(path)
    for name in ("gps", "gpsStart", "gpsCentroid", "gpsPeak"):
        assert frame[name].dtype == np.float64
    for name in ("EnWDF", "sigma", "snrPeak", "tSpread", "duration",
                 "freqMin", "freqMean", "freqMax"):
        assert frame[name].dtype == np.float32


def test_the_measurement_columns_are_written_so_they_compress(tmp_path):
    """The file is mostly columns of measurements, whose mantissas do not
    compress on their own. Losing these settings silently costs about a factor
    of two, which no test of the values would notice.
    """
    import pyarrow.parquet as pq

    rng = np.random.default_rng(6)
    path = str(tmp_path / "triggers.parquet")
    with TriggerWriter(path) as writer:
        for k in range(200):
            writer.append(trigger_record(1.4e9 + k, *sparse_window(512, rng), 512))

    metadata = pq.ParquetFile(path).metadata
    encodings = {}
    for column in range(metadata.num_columns):
        chunk = metadata.row_group(0).column(column)
        encodings[chunk.path_in_schema] = set(chunk.encodings)

    assert "BYTE_STREAM_SPLIT" in encodings["EnWDF"]
    assert "BYTE_STREAM_SPLIT" in encodings["gps"]
    # The basis name is a small set of repeated strings, which is what
    # dictionary encoding is for.
    assert "RLE_DICTIONARY" in encodings["wave"] or "PLAIN_DICTIONARY" in encodings["wave"]
    assert metadata.row_group(0).column(0).compression == "ZSTD"
