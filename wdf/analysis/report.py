"""Self-contained per-run HTML report (glitchgram, winning-wavelet-basis
distribution, top-N loudest events, time-frequency plot of the loudest
survivor, one section per detector),
in the spirit of coherent WaveBurst's own per-run HTML report pages: a single
file, no external assets (all plots embedded as base64 PNGs), safe to open
offline or attach anywhere. Meant to be called once at the end of a
notebook/script's run, alongside its own analysis, not as a replacement for
either -- this is a fixed, general-purpose diagnostic dump, not a substitute
for a specific investigation's own plots.

Deliberately reuses matplotlib (already a wdflow dependency) rather than
building a hand-rolled charting layer -- keeps this module small and its
output easy to extend without a second charting convention to maintain.
"""
from __future__ import annotations

import base64
import io
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wdf.analysis.wavelets import wavelet_coeff_tiles


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _wave_stats_b64(ifo: str, df: pd.DataFrame) -> str | None:
    """Bar chart of how often each wavelet basis won `WDF2Classify`'s
    per-window basis selection (the `wave` trigger column) -- useful to see,
    on real data, whether a handful of bases dominate or the winning basis
    is spread roughly evenly across the whole candidate list (see the
    wavelet-basis-count question this was built for: cutting p4TSA's
    candidate-basis list down needs this kind of evidence, not a guess).
    """
    if "wave" not in df.columns or df["wave"].empty:
        return None
    counts = df["wave"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.28 * len(counts))), dpi=110)
    ax.barh(counts.index, counts.values, color="steelblue")
    ax.set_xlabel("n triggers")
    ax.set_title(f"{ifo}: winning-basis distribution ({len(df)} triggers, "
                 f"{df['wave'].nunique()} distinct bases)")
    for y, v in enumerate(counts.values):
        ax.text(v, y, f" {v} ({100 * v / len(df):.1f}%)", va="center", fontsize=7)
    return _fig_to_b64(fig)


def _glitchgram_b64(ifo: str, df: pd.DataFrame, gps_reference: float | None) -> str:
    fig, ax = plt.subplots(figsize=(9, 4), dpi=110)
    x = df["gpsPeak"] - gps_reference if gps_reference is not None else df["gpsPeak"]
    sc = ax.scatter(x, df["freqPeak"], c=np.log10(df["snrPeak"].clip(lower=1e-3)),
                     cmap="viridis", s=10, alpha=0.6, linewidths=0)
    if gps_reference is not None:
        ax.axvline(0.0, color="crimson", ls="--", lw=1, label="reference GPS")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel(f"time - {gps_reference:.1f} [s]")
    else:
        ax.set_xlabel("gpsPeak [s]")
    ax.set_ylabel("freqPeak [Hz]")
    ax.set_title(f"{ifo}: glitchgram ({len(df)} triggers)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log10(snrPeak)")
    return _fig_to_b64(fig)


def _tf_plot_b64(ifo: str, member_rows: pd.DataFrame, fs: float, sigma: float,
                  center_gps: float, snr_label: float) -> str | None:
    wt_cols = sorted((c for c in member_rows.columns if c.startswith("wt") and c[2:].isdigit()),
                      key=lambda c: int(c[2:]))
    if not wt_cols:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    any_tile = False
    for _, trig in member_rows.iterrows():
        wt = trig[wt_cols].to_numpy(dtype=float)
        gps0 = trig["gps"]
        for t_lo, t_hi, f_lo, f_hi, mag in wavelet_coeff_tiles(wt, fs):
            if mag <= 0:
                continue
            any_tile = True
            ax.add_patch(plt.Rectangle(
                (gps0 + t_lo - center_gps, f_lo), t_hi - t_lo, f_hi - f_lo,
                color=plt.cm.viridis(min(mag / (10 * sigma), 1.0)), alpha=0.8,
            ))
    if not any_tile:
        plt.close(fig)
        return None
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(0, fs / 2)
    ax.set_xlabel(f"time - {center_gps:.3f} [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(f"{ifo}: time-frequency tiles, loudest survivor (EnWDF={snr_label:.1f})")
    return _fig_to_b64(fig)


def _top_n_table_html(top: pd.DataFrame, columns: list[str]) -> str:
    header = "".join(f"<th>{c}</th>" for c in columns)
    rows = []
    for _, row in top.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            cells.append(f"<td>{v:.3f}</td>" if isinstance(v, (int, float, np.floating)) else f"<td>{v}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_run_report(
    outdir: str,
    cleaned: dict[str, pd.DataFrame],
    clustered: dict[str, pd.DataFrame] | None = None,
    raw_triggers: dict[str, pd.DataFrame] | None = None,
    par: dict | None = None,
    gps_reference: float | None = None,
    event_name: str = "run",
    top_n: int = 5,
    rank_col: str | None = None,
    filename: str = "report.html",
) -> str:
    """Writes a self-contained HTML report to `outdir/filename` and returns
    its path.

    :type outdir: str
    :param outdir: directory to write the report into (created if missing).
    :type cleaned: dict[str, pandas.DataFrame]
    :param cleaned: {ifo: cleaned raw-trigger DataFrame} (e.g.
        `wdf.analysis.io.clean_triggers`'s output) -- drives the glitchgram.
        Must have `gpsPeak`, `freqPeak`, `snrPeak` columns.
    :type clustered: dict[str, pandas.DataFrame] | None
    :param clustered: {ifo: clustered-event DataFrame} (e.g.
        `wdf.analysis.clustering.TriggerClusterer.clustered_events`'s output)
        -- drives the top-N table. Falls back to `cleaned` (per-trigger, not
        per-cluster) if omitted.
    :type raw_triggers: dict[str, pandas.DataFrame] | None
    :param raw_triggers: {ifo: raw-trigger DataFrame with `wt*` columns}
        (`fullPrint >= 1`) -- drives the time-frequency plot of each
        detector's loudest surviving event. Skipped per-detector if omitted
        or missing `wt*` columns.
    :type par: dict | None
    :param par: {ifo: run Parameters-like object} with `.resampling`/`.sigma`
        -- needed for the time-frequency plot's frequency axis/color scale.
        Skipped per-detector if omitted.
    :type gps_reference: float | None
    :param gps_reference: if given, the glitchgram's time axis is expressed
        relative to this GPS (e.g. a known event's merger time), with a
        marker line at zero. Otherwise absolute `gpsPeak` is used.
    :type event_name: str
    :param event_name: label shown in the report's title.
    :type top_n: int
    :param top_n: how many loudest events to list per detector.
    :type rank_col: str | None
    :param rank_col: column used to rank/select the "loudest" events. Default
        (None): `EnWDF` if `clustered` is given (or any per-detector
        DataFrame already has it), else `snrPeak` (the raw-trigger schema
        `cleaned` falls back to when `clustered` is omitted).
    :type filename: str
    :param filename: output file name within `outdir`.
    :return: str -- the full path to the written HTML file.
    """
    os.makedirs(outdir, exist_ok=True)
    clustered = clustered or cleaned
    raw_triggers = raw_triggers or {}
    par = par or {}

    sections = []
    for ifo, df in cleaned.items():
        glitchgram_b64 = _glitchgram_b64(ifo, df, gps_reference)
        wave_stats_b64 = _wave_stats_b64(ifo, df)
        wave_html = ""
        if wave_stats_b64:
            wave_html = (f'<figure><img src="data:image/png;base64,{wave_stats_b64}" alt="{ifo} wave stats">'
                         f"<figcaption>Winning wavelet basis distribution</figcaption></figure>")

        ranking_df = clustered.get(ifo, df)
        if rank_col is not None:
            this_rank_col = rank_col
        elif "EnWDF" in ranking_df.columns:
            this_rank_col = "EnWDF"
        else:
            this_rank_col = "snrPeak"
        top = ranking_df.sort_values(this_rank_col, ascending=False).head(top_n)
        table_cols = [c for c in ("gpsStart", "gpsPeak", this_rank_col, "freqMax", "freqPeak",
                                   "n_triggers") if c in top.columns]
        if this_rank_col not in table_cols:
            table_cols.append(this_rank_col)
        table_html = _top_n_table_html(top, table_cols)

        tf_html = ""
        if ifo in raw_triggers and ifo in par and len(top):
            best = top.iloc[0]
            center_gps = best["gpsPeak"]
            raw = raw_triggers[ifo]
            window_s = 0.5
            near = raw[(raw["gps"] - center_gps).abs() <= window_s] if "gps" in raw.columns else raw.iloc[0:0]
            tf_b64 = _tf_plot_b64(ifo, near, float(par[ifo].resampling), float(par[ifo].sigma),
                                   float(center_gps), float(best[this_rank_col]))
            if tf_b64:
                tf_html = (f'<figure><img src="data:image/png;base64,{tf_b64}" alt="{ifo} TF">'
                           f"<figcaption>Time-frequency tiles, loudest survivor</figcaption></figure>")

        sections.append(f"""
<section class="detector">
  <h2>{ifo}</h2>
  <div class="plots">
    <figure><img src="data:image/png;base64,{glitchgram_b64}" alt="{ifo} glitchgram">
      <figcaption>Glitchgram</figcaption></figure>
    {wave_html}
    {tf_html}
  </div>
  <h3>Top {top_n} by {this_rank_col}</h3>
  <div class="table-wrap">{table_html}</div>
</section>
""")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{event_name} -- run report</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1100px; margin: 0 auto; padding: 24px; }}
h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
.plots {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.plots figure {{ margin: 0; flex: 1 1 420px; max-width: 100%; }}
.plots img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
figcaption {{ font-size: 0.82rem; color: #666; margin-top: 4px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
th {{ color: #666; font-weight: 600; }}
</style></head>
<body>
<h1>{event_name} -- run report</h1>
{''.join(sections)}
</body></html>
"""
    path = os.path.join(outdir, filename)
    with open(path, "w") as fh:
        fh.write(html)
    return path
