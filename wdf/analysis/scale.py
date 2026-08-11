"""The analysis window length as a third coordinate, beside time and frequency.

A run searches at one window length. The schedule accepts several, because the
length is a parameter of the search rather than a property of it, and this
module is what makes the results comparable when more than one is used.

Several lengths are not several catalogues to be merged, nor several energies to
be added: they are tilings of one signal, strongly correlated, since every one
of them is computed from the same data. Adding them would count one transient's
energy several times.

They are made comparable instead of combined. The search statistic is not the
same quantity at two window lengths -- the Donoho-Johnstone threshold deciding
which coefficients survive depends on how many the window holds -- so each
window length's energy is mapped through its own background distribution to a
probability, and from there to a significance

    S = -log P(E' >= E | H0, scale, band)

which means the same thing at every window length. That is the variable the
pixel cloud carries, and the one a cross-scale comparison can be made on.

The cloud is kept three-dimensional, `(t, f, scale)`, so that the length stays a
coordinate to be measured rather than something averaged over. Under a run at
one length the third coordinate is degenerate and everything below reduces to
the single-scale case.

`scale_maximum` forms the first cross-scale statistic, the maximum of S over the
window lengths that can represent a given tile. Its own distribution is not the
distribution of a single window's S: taking a maximum over three correlated
searches is a look-elsewhere effect, and the resulting statistic has to be
calibrated on the background in its own right, which `ScaleCalibration.fit`
makes possible for it exactly as for a single scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wdf.analysis.wavelets import coeff_freq_bands, coeff_time_bounds

SCALE_PIXEL_COLUMNS = [
    "trigger_index", "ifo", "scale", "fs",
    "t_lo", "t_hi", "f_lo", "f_hi", "energy", "sigma",
]


def pixel_cloud(triggers: pd.DataFrame) -> pd.DataFrame:
    """Every surviving coefficient of every window length, as one tile cloud.

    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `gps`, `n_coeff`, `fs` and the
        coefficient columns. Several analysis window lengths may be present.
    :return: pandas.DataFrame -- one row per tile, with `SCALE_PIXEL_COLUMNS`.
        `scale` is the window length the tile was found at, in samples; times
        are absolute GPS.
    """
    if triggers.empty or "wt_index" not in triggers:
        return pd.DataFrame(columns=SCALE_PIXEL_COLUMNS)

    frames = []
    for (n_coeff, fs), group in triggers.groupby(["n_coeff", "fs"], sort=True):
        n_coeff, fs = int(n_coeff), float(fs)
        t_lo_of, t_hi_of = coeff_time_bounds(n_coeff, fs)
        f_lo_of, f_hi_of = coeff_freq_bands(n_coeff, fs)

        counts = group["wt_index"].map(len).to_numpy()
        if counts.sum() == 0:
            continue
        index = np.concatenate([np.asarray(v, dtype=int)
                                for v in group["wt_index"].to_numpy()])
        value = np.concatenate([np.asarray(v, dtype=float)
                                for v in group["wt_value"].to_numpy()])
        gps = np.repeat(group["gps"].to_numpy(dtype=float), counts)

        frames.append(pd.DataFrame(dict(
            trigger_index=np.repeat(group.index.to_numpy(), counts),
            ifo=(np.repeat(group["ifo"].to_numpy(), counts)
                 if "ifo" in group else np.full(len(index), "")),
            scale=n_coeff,
            fs=fs,
            t_lo=gps + t_lo_of[index],
            t_hi=gps + t_hi_of[index],
            f_lo=f_lo_of[index],
            f_hi=f_hi_of[index],
            energy=value ** 2,
            sigma=np.repeat(group["sigma"].to_numpy(dtype=float), counts),
        )))
    if not frames:
        return pd.DataFrame(columns=SCALE_PIXEL_COLUMNS)
    return pd.concat(frames, ignore_index=True)[SCALE_PIXEL_COLUMNS]


def normalised_energy(pixels: pd.DataFrame) -> np.ndarray:
    """Each tile's energy on its own window's noise scale.

    :type pixels: pandas.DataFrame
    :param pixels: tiles carrying `energy` and `sigma`.
    :return: numpy.ndarray -- the energy in units of the noise variance.
    """
    sigma = pixels["sigma"].to_numpy(dtype=float)
    energy = pixels["energy"].to_numpy(dtype=float)
    valid = np.isfinite(sigma) & (sigma > 0.0)
    return np.divide(energy, sigma ** 2, out=np.full(len(pixels), np.nan),
                     where=valid)


@dataclass
class ScaleCalibration:
    """The background energy distribution of every (window length, band).

    A tile's significance is read from the distribution of its own window
    length and its own frequency band, so that the same number means the same
    thing wherever it was measured. Bands are octaves of the dyadic ladder and
    are shared between window lengths wherever both reach them, so the key is
    exact rather than a binning choice.
    """

    tables: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, background: pd.DataFrame) -> "ScaleCalibration":
        """Measure the background distribution from injection-free data.

        :type background: pandas.DataFrame
        :param background: a pixel cloud from data containing no signal.
        :return: ScaleCalibration
        :raises ValueError: if the cloud is empty.
        """
        if background.empty:
            raise ValueError(
                "the background pixel cloud is empty: a significance is read "
                "from a measured distribution, and there is nothing to measure"
            )
        energy = normalised_energy(background)
        tables = {}
        keys = list(zip(background["scale"].to_numpy(dtype=int),
                        background["f_lo"].to_numpy(dtype=float)))
        frame = pd.DataFrame(dict(key=pd.Series(keys, dtype=object), energy=energy))
        for key, group in frame.dropna().groupby("key", sort=False):
            tables[key] = np.sort(group["energy"].to_numpy())
        return cls(tables=tables)

    def significance(self, pixels: pd.DataFrame) -> np.ndarray:
        """`-log P(E' >= E | H0, scale, band)` for every tile.

        A tile whose (window length, band) the background never produced has no
        measured distribution and is returned as NaN rather than assigned one.

        :type pixels: pandas.DataFrame
        :param pixels: a pixel cloud.
        :return: numpy.ndarray -- the significance of each tile, in nats.
        """
        energy = normalised_energy(pixels)
        out = np.full(len(pixels), np.nan)
        scale = pixels["scale"].to_numpy(dtype=int)
        band = pixels["f_lo"].to_numpy(dtype=float)
        for key, table in self.tables.items():
            rows = np.flatnonzero((scale == key[0]) & (band == key[1]))
            if rows.size == 0:
                continue
            above = len(table) - np.searchsorted(table, energy[rows], side="left")
            out[rows] = -np.log((above + 1.0) / (len(table) + 1.0))
        return out


def scale_maximum(pixels: pd.DataFrame,
                  calibration: ScaleCalibration) -> pd.DataFrame:
    """The cross-scale statistic and the scale signature of every tile.

    Each tile is evaluated at every window length that reaches its band: at
    each, the tile containing its centre carries a significance, or none if no
    coefficient there survived thresholding. The maximum over window lengths is
    the cross-scale statistic; how many of them saw the tile at all, and how
    evenly, describe the transient's signature across resolutions.

    :type pixels: pandas.DataFrame
    :param pixels: a pixel cloud, from one or several window lengths.
    :type calibration: ScaleCalibration
    :param calibration: the background distributions to read significances from.
    :return: pandas.DataFrame -- `pixels` with `significance` (its own scale's),
        `s_max` (the maximum over scales), `scale_best` (the window length
        attaining it), `n_scales` (how many saw it) and `s_ratio` (the smallest
        significance over the largest, among the scales that saw it).
    """
    out = pixels.copy().reset_index(drop=True)
    if out.empty:
        return out.assign(significance=[], s_max=[], scale_best=[],
                          n_scales=[], s_ratio=[])

    own = calibration.significance(out)
    out["significance"] = own

    t_mid = 0.5 * (out["t_lo"].to_numpy(dtype=float)
                   + out["t_hi"].to_numpy(dtype=float))
    scale = out["scale"].to_numpy(dtype=int)
    band = out["f_lo"].to_numpy(dtype=float)

    # Within one window length and one band the tiles partition time exactly,
    # so the tile covering an instant is found by searching the band's own
    # sorted edges rather than by testing every tile.
    lookup = {}
    for key, rows in out.groupby(["scale", "f_lo"], sort=False).indices.items():
        rows = np.asarray(rows)
        order = rows[np.argsort(out["t_lo"].to_numpy(dtype=float)[rows])]
        lookup[(int(key[0]), float(key[1]))] = (
            out["t_lo"].to_numpy(dtype=float)[order],
            out["t_hi"].to_numpy(dtype=float)[order],
            own[order],
        )

    scales = np.unique(scale)
    reachable = np.zeros((len(out), len(scales)))
    seen = np.zeros((len(out), len(scales)), dtype=bool)
    for column, other in enumerate(scales):
        for this_band in np.unique(band):
            key = (int(other), float(this_band))
            if key not in lookup:
                continue
            rows = np.flatnonzero(band == this_band)
            t_lo, t_hi, values = lookup[key]
            slot = np.searchsorted(t_lo, t_mid[rows], side="right") - 1
            inside = (slot >= 0) & (t_mid[rows] < t_hi[np.maximum(slot, 0)])
            found = rows[inside]
            reachable[found, column] = np.nan_to_num(values[slot[inside]])
            seen[found, column] = True

    out["s_max"] = reachable.max(axis=1)
    out["scale_best"] = scales[reachable.argmax(axis=1)]
    out["n_scales"] = seen.sum(axis=1)
    smallest = np.where(seen, reachable, np.inf).min(axis=1)
    out["s_ratio"] = np.divide(smallest, out["s_max"],
                               out=np.zeros(len(out)),
                               where=out["s_max"].to_numpy() > 0)
    return out
