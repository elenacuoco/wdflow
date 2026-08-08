"""The per-cluster coefficient matrix, which the GNN coincidence, the waveform
reconstruction and the parameter estimation all read."""
import numpy as np
import pandas as pd
import pytest

from _synth import triggers_from_signal
from wdf.analysis.cluster_coefficients import (ClusterCoefficients,
                                               collect_cluster_coefficients)

FS, WINDOW, OVERLAP, WAVE, SIGMA = 2048.0, 512, 128, "DaubC12", 1.0
STEP = WINDOW - OVERLAP


def triggers_from(signal, gps0=1000.0, ifo="H1"):
    return triggers_from_signal(signal, FS, WINDOW, OVERLAP, gps0=gps0, ifo=ifo,
                                wave=WAVE, sigma=SIGMA)


def a_signal(n_windows=4, seed=0):
    rng = np.random.default_rng(seed)
    length = WINDOW + (n_windows - 1) * STEP
    envelope = np.hanning(length)
    phase = 2 * np.pi * np.cumsum(np.linspace(80.0, 300.0, length)) / FS
    return 5.0 * envelope * np.sin(phase) + 1e-3 * rng.standard_normal(length)


def test_matrix_has_one_row_per_window_and_one_column_per_coefficient():
    triggers = triggers_from(a_signal(n_windows=4))
    cluster = ClusterCoefficients.from_triggers(triggers, FS, WINDOW, OVERLAP)

    assert cluster.coefficients.shape == (len(triggers), WINDOW)
    assert cluster.n_triggers == len(triggers)
    assert cluster.n_coeff == WINDOW
    assert cluster.ifo == "H1"
    assert np.all(np.diff(cluster.times) > 0)


def test_a_single_window_transient_is_a_one_row_matrix():
    """A singleton is a cluster with one row, not a missing one."""
    triggers = triggers_from(a_signal(n_windows=1))
    cluster = ClusterCoefficients.from_triggers(triggers, FS, WINDOW, OVERLAP)

    assert cluster.n_triggers == 1
    assert cluster.coefficients.shape == (1, WINDOW)
    assert cluster.duration == pytest.approx(WINDOW / FS)


def test_enwdf_of_the_cluster_exceeds_its_loudest_window():
    """The statistic is the stitched reconstruction, so it covers the whole
    signal rather than the best single window."""
    triggers = triggers_from(a_signal(n_windows=5))
    cluster = ClusterCoefficients.from_triggers(triggers, FS, WINDOW, OVERLAP)

    assert cluster.enwdf() > triggers.EnWDF.max()


def test_reconstruction_round_trips_through_the_frame():
    triggers = triggers_from(a_signal(n_windows=3))
    cluster = ClusterCoefficients.from_triggers(triggers, FS, WINDOW, OVERLAP)

    gps_start, samples = cluster.reconstruct()
    assert gps_start == pytest.approx(cluster.gps_start)
    assert samples.size == int(round(cluster.duration * FS))


@pytest.mark.parametrize("n_windows", [1, 3, 7])
def test_the_wavegram_has_the_same_shape_whatever_the_cluster(n_windows):
    """A graph node needs a fixed-width feature vector, however many windows
    the transient happens to span."""
    triggers = triggers_from(a_signal(n_windows=n_windows))
    cluster = ClusterCoefficients.from_triggers(triggers, FS, WINDOW, OVERLAP)

    grid = cluster.wavegram(n_time_bins=32)
    assert grid.shape == (int(np.log2(WINDOW)) + 1, 32)
    assert np.all(grid >= 0.0)
    assert grid.max() > 0.0


def test_the_wavegram_is_on_the_noise_scale():
    """Cells hold |coefficient| / sigma, so two detectors are comparable."""
    triggers = triggers_from(a_signal(n_windows=3))
    loud = ClusterCoefficients.from_triggers(triggers, FS, WINDOW, OVERLAP)
    quiet = ClusterCoefficients.from_triggers(
        triggers.assign(sigma=2.0), FS, WINDOW, OVERLAP)

    assert quiet.wavegram().max() == pytest.approx(loud.wavegram().max() / 2.0)


def test_the_wavegram_shape_survives_a_change_of_amplitude():
    """The unit-normalised grid is what the cross-detector edge compares, so a
    signal seen at half the amplitude must still look like the same signal."""
    triggers = triggers_from(a_signal(n_windows=4))
    halved = triggers.copy()
    halved["wt_value"] = [0.5 * np.asarray(v) for v in halved["wt_value"]]

    a = ClusterCoefficients.from_triggers(triggers, FS, WINDOW, OVERLAP).wavegram().ravel()
    b = ClusterCoefficients.from_triggers(halved, FS, WINDOW, OVERLAP).wavegram().ravel()
    a, b = np.log1p(a), np.log1p(b)

    similarity = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert similarity > 0.99


def test_two_different_transients_are_less_alike_than_one_with_itself():
    same = a_signal(n_windows=4, seed=1)
    rng = np.random.default_rng(2)
    other = np.zeros_like(same)
    other[: WINDOW] = 5.0 * rng.standard_normal(WINDOW)

    def grid(signal):
        cluster = ClusterCoefficients.from_triggers(
            triggers_from(signal), FS, WINDOW, OVERLAP)
        g = np.log1p(cluster.wavegram().ravel())
        return g / np.linalg.norm(g)

    a, b = grid(same), grid(other)
    assert float(a @ a) == pytest.approx(1.0)
    assert float(a @ b) < 0.9


def test_collect_builds_one_matrix_per_event_including_singletons():
    triggers = pd.concat([triggers_from(a_signal(n_windows=3), gps0=1000.0),
                          triggers_from(a_signal(n_windows=1), gps0=2000.0)],
                         ignore_index=True)
    labeled = triggers.assign(cluster_id=[0, 0, 0, 1])
    events = pd.DataFrame({"cluster_id": [0, 1], "n_triggers": [3, 1]})

    per_cluster = collect_cluster_coefficients(labeled, events, FS, WINDOW, OVERLAP)

    assert set(per_cluster) == {0, 1}
    assert per_cluster[0].n_triggers == 3
    assert per_cluster[1].n_triggers == 1
    assert per_cluster[0].wavegram().shape == per_cluster[1].wavegram().shape


def test_scoring_on_the_reconstruction_recovers_what_the_windows_split():
    """A transient longer than the window fails a threshold in every single
    window while its whole reconstruction passes it -- which is the reason to
    threshold on the event rather than on the window."""
    from wdf.analysis.cluster_coefficients import score_events_by_reconstruction

    triggers = triggers_from(a_signal(n_windows=6, seed=4))
    cluster = ClusterCoefficients.from_triggers(triggers, FS, WINDOW, OVERLAP,
                                                cluster_id=0)
    events = pd.DataFrame({"cluster_id": [0], "EnWDF": [triggers.EnWDF.max()]})

    scored = score_events_by_reconstruction(events, {0: cluster})

    assert scored.loc[0, "EnWDF_window"] == pytest.approx(triggers.EnWDF.max())
    assert scored.loc[0, "n_windows"] == len(triggers)
    assert scored.loc[0, "EnWDF"] > scored.loc[0, "EnWDF_window"]

    # a threshold between the two admits the event and rejects every window
    threshold = 0.5 * (scored.loc[0, "EnWDF"] + scored.loc[0, "EnWDF_window"])
    assert (triggers.EnWDF < threshold).all()
    assert scored.loc[0, "EnWDF"] > threshold


def test_a_single_window_event_is_unchanged_by_the_rescoring():
    """The control: a burst that fits in one window has nothing to stitch."""
    from wdf.analysis.cluster_coefficients import score_events_by_reconstruction

    triggers = triggers_from(a_signal(n_windows=1, seed=5))
    cluster = ClusterCoefficients.from_triggers(triggers, FS, WINDOW, OVERLAP,
                                                cluster_id=0)
    events = pd.DataFrame({"cluster_id": [0], "EnWDF": [triggers.EnWDF.max()]})

    scored = score_events_by_reconstruction(events, {0: cluster})
    assert scored.loc[0, "EnWDF"] == pytest.approx(scored.loc[0, "EnWDF_window"], rel=1e-6)


def test_an_event_without_coefficients_keeps_its_per_window_value():
    from wdf.analysis.cluster_coefficients import score_events_by_reconstruction

    events = pd.DataFrame({"cluster_id": [7], "EnWDF": [9.5]})
    scored = score_events_by_reconstruction(events, {})
    assert scored.loc[0, "EnWDF"] == pytest.approx(9.5)
    assert scored.loc[0, "n_windows"] == 0


def test_the_matrix_is_stored_in_single_precision():
    """A segment holds hundreds of thousands of these; double precision costs
    gigabytes for digits nothing reads."""
    triggers = triggers_from(a_signal(n_windows=2))
    cluster = ClusterCoefficients.from_triggers(triggers, FS, WINDOW, OVERLAP)

    assert cluster.coefficients.dtype == np.float32
    # the reconstruction still goes through double precision
    _, samples = cluster.reconstruct()
    assert samples.dtype == np.float64


def test_the_stream_scores_the_same_as_the_mapping_without_holding_it():
    """`iter_cluster_coefficients` exists so a whole segment need not be built
    before deciding which events matter."""
    from wdf.analysis.cluster_coefficients import (iter_cluster_coefficients,
                                                   score_events_by_reconstruction)

    triggers = pd.concat([triggers_from(a_signal(n_windows=3), gps0=1000.0),
                          triggers_from(a_signal(n_windows=2), gps0=2000.0)],
                         ignore_index=True)
    labeled = triggers.assign(cluster_id=[0, 0, 0, 1, 1])
    events = pd.DataFrame({"cluster_id": [0, 1],
                           "EnWDF": [triggers.EnWDF.max(), triggers.EnWDF.min()]})

    mapping = collect_cluster_coefficients(labeled, events, FS, WINDOW, OVERLAP)
    from_mapping = score_events_by_reconstruction(events, mapping)
    from_stream = score_events_by_reconstruction(
        events, iter_cluster_coefficients(labeled, events, FS, WINDOW, OVERLAP))

    pd.testing.assert_frame_equal(from_mapping, from_stream)

    # the stream yields lazily: taking one costs one cluster, not the segment
    stream = iter_cluster_coefficients(labeled, events, FS, WINDOW, OVERLAP)
    label, first = next(stream)
    assert label == 0 and first.n_triggers == 3
