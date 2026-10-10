"""GlitchAtomsFrom: the online worker on samples cleaned by an offline run's M4 atoms."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("py4tsa")

from test_online_worker import FS, SECONDS, T0, parameters, strain  # noqa: E402

from wdf.processes.glitch_subtraction import Atom, GlitchModel  # noqa: E402
from wdf.processes.wdfUnitOnline import ArraySource, wdfUnitOnline  # noqa: E402

START, END = T0 + 26.0, T0 + SECONDS
RATE = FS / 2
CUT = T0 + 120.0
MODES = {"causal": {}, "zerophase": dict(WhiteningMode="zerophase", ZeroPhaseDuration=8.0)}
COLUMNS = ["centre", "frequency", "q", "alpha", "beta", "snr", "rate", "half_length"]


def atoms(centres, rate=RATE, amplitude=30.0):
    """Loud sine-Gaussians in the conditioned domain, as GlitchAtoms rows."""
    out = []
    for k, c in enumerate(centres):
        f, q = 60.0 + 50.0 * k, 8.0
        tau = q / (2 * np.pi * f) / np.sqrt(2)
        out.append(Atom(centre=c, frequency=f, q=q, alpha=amplitude, beta=0.5 * amplitude, snr=50.0,
                        rate=rate, half_length=int(np.ceil(5.0 * tau * rate))))
    return out


def write(path, rows):
    pd.DataFrame([vars(a) for a in rows], columns=COLUMNS).to_csv(path, index=False)
    return str(path)


class Recording(wdfUnitOnline):
    """The worker, keeping its conditioned and whitened samples."""

    def _condition(self, ds, raw, t):
        y, start = super()._condition(ds, raw, t)
        if y is not None:
            self.conditioned.append((start, np.array(y, dtype=float)))
        return y, start

    def _consume(self, w, w_start):
        self.whitened.append((float(w_start), np.array(w, dtype=float)))
        super()._consume(w, w_start)

    def process(self, source, gps_start, gps_end):
        self.conditioned, self.whitened = [], []
        return super().process(source, gps_start, gps_end)


class Reference(Recording):
    """The subtraction done outside the worker, on the conditioned samples it
    releases, as a caller patching `_condition` would."""

    model = None

    def _condition(self, ds, raw, t):
        y, start = wdfUnitOnline._condition(self, ds, raw, t)
        if y is not None:
            m = self.model.evaluate(start, float(self.par.resampling), len(y))
            if m is not None:
                y = np.asarray(y, float) - m
            self.conditioned.append((start, np.array(y, dtype=float)))
        return y, start


def stream(chunks):
    return np.concatenate([c for _, c in chunks])


def run(cls, x=None, **kw):
    worker = cls(parameters(**kw))
    result = worker.process(ArraySource(strain() if x is None else x, T0, FS), START, END)
    return worker, result


@pytest.fixture(scope="module", params=sorted(MODES))
def runs(request, tmp_path_factory):
    kw = MODES[request.param]
    folder = tmp_path_factory.mktemp(request.param)
    near = atoms([T0 + 60.0, T0 + 100.0, T0 + 140.0])
    path = write(folder / "GlitchAtoms-Searchoffline.csv", near)
    far = write(folder / "far.csv", atoms([T0 + 5000.0]))
    plain, plain_result = run(Recording, **kw)
    cleaned, cleaned_result = run(Recording, GlitchAtomsFrom=path, **kw)
    Reference.model = GlitchModel(near)
    reference, reference_result = run(Reference, **kw)
    return dict(kw=kw, path=path, far=far, near=near, plain=plain, plain_result=plain_result,
                cleaned=cleaned, cleaned_result=cleaned_result, reference=reference,
                reference_result=reference_result)


def test_unset_or_far_atoms_change_nothing(runs):
    """Unset, empty, or atoms that reach no sample: bit-identical outputs."""
    kw, plain = runs["kw"], runs["plain"]
    every = runs["plain_result"].before(END + 1e6)
    for value in ("", runs["far"]):
        other, result = run(Recording, GlitchAtomsFrom=value, **kw)
        np.testing.assert_array_equal(stream(other.conditioned), stream(plain.conditioned))
        np.testing.assert_array_equal(stream(other.whitened), stream(plain.whitened))
        assert result.before(END + 1e6) == every
    assert plain.par.GlitchAtomsFrom == "" and plain.par.OnlineRun is True
    assert not hasattr(plain.par, "GlitchAtomsLoaded")


def test_the_atoms_are_subtracted_from_the_conditioned_samples(runs):
    """Each conditioned block is the plain one minus the atoms evaluated on it."""
    plain, cleaned = runs["plain"], runs["cleaned"]
    model = GlitchModel(runs["near"])
    hit = 0
    assert [s for s, _ in plain.conditioned] == [s for s, _ in cleaned.conditioned]
    for (start, y0), (_, y1) in zip(plain.conditioned, cleaned.conditioned):
        m = model.evaluate(start, RATE, y0.size)
        if m is None:
            np.testing.assert_array_equal(y1, y0)
        else:
            hit += 1
            np.testing.assert_array_equal(y1, y0 - m)
    assert hit == cleaned.par.GlitchAtomsBlocks >= 3
    assert not np.array_equal(stream(plain.whitened), stream(cleaned.whitened))


def test_the_parameter_equals_the_reference_subtraction(runs):
    """The run with the parameter and the run cleaned outside the worker give the
    same conditioned and whitened samples and the same outputs."""
    cleaned, reference = runs["cleaned"], runs["reference"]
    np.testing.assert_array_equal(stream(cleaned.conditioned), stream(reference.conditioned))
    np.testing.assert_array_equal(stream(cleaned.whitened), stream(reference.whitened))
    assert runs["cleaned_result"].before(END + 1e6) == runs["reference_result"].before(END + 1e6)


def test_the_run_records_that_it_is_not_online(runs, tmp_path):
    par = runs["cleaned"].par
    assert par.GlitchAtomsFrom == runs["path"] and par.OnlineRun is False
    assert par.GlitchAtomsLoaded == 3
    worker = wdfUnitOnline(parameters(GlitchAtomsFrom=runs["path"], outdir=str(tmp_path) + "/",
                                      run="atoms", **runs["kw"]))
    worker.segmentProcess((START, T0 + 60.0), ArraySource(strain(), T0, FS))
    used = json.load(open(next(tmp_path.glob("atoms/H1/*/parametersUsed-Online*.json"))))
    assert used["GlitchAtomsFrom"] == runs["path"] and used["OnlineRun"] is False
    assert used["GlitchAtomsLoaded"] == 3


def test_the_latency_is_unchanged_beyond_the_atoms(runs, tmp_path):
    """With the same atoms file, altering the input after t changes no output
    emitted at or before t (causal) or of data time at or before t - Latency_s
    (zero phase); and an atom that starts after t changes none of them either."""
    kw = runs["kw"]
    base = runs["cleaned_result"]
    x = strain()
    i = int(round((CUT - T0) * FS))
    x[i:] = np.random.default_rng(9).standard_normal(x.size - i) * 3.0
    _, altered = run(Recording, x=x, GlitchAtomsFrom=runs["path"], **kw)
    late = runs["near"] + atoms([CUT + 10.0], amplitude=60.0)
    assert late[-1].span[0] > CUT
    _, later = run(Recording, GlitchAtomsFrom=write(tmp_path / "late.csv", late), **kw)
    lag = base.parameters.Latency_s if kw else 0.0
    for other in (altered, later):
        assert base.before(CUT) == other.before(CUT)
        assert base.before(END + 1e6) != other.before(END + 1e6)
    if kw:
        assert all(r["data_time"] <= CUT - lag for r in base.before(CUT)["triggers"])


def test_atoms_at_another_rate_are_refused(tmp_path):
    path = write(tmp_path / "atoms.csv", atoms([T0 + 60.0], rate=4096.0))
    with pytest.raises(ValueError, match="GlitchAtomsFrom .* 4096.0 Hz"):
        run(Recording, GlitchAtomsFrom=path)
