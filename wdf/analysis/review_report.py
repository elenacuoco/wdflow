"""Self-contained review report for a search run against known injections."""
from __future__ import annotations

import base64
import io
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from wdf.analysis.injections import efficiency

C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def build_review_report(outdir, recovery, trigger_summary=None, coincidence_matched=None,
                        candidates=None, background_candidates=None,
                        false_alarm_candidates=None, injections=None, livetime_days=None,
                        title="WDF review on simulated data", basename="review_report"):
    """Write an HTML and a Markdown summary of a run against known injections.

    :type outdir: str
    :param outdir: directory to write into; created if missing.
    :type recovery: pandas.DataFrame
    :param recovery: per-detector `match_injections` output, with `injected_snr`.
    :type trigger_summary: pandas.DataFrame | None
    :param trigger_summary: per-detector, per-frame trigger and cluster counts.
    :type coincidence_matched: pandas.DataFrame | None
    :param coincidence_matched: `match_injections` output for coincident candidates.
    :type candidates: pandas.DataFrame | None
    :param candidates: coincidences found in the foreground frame.
    :type background_candidates: pandas.DataFrame | None
    :param background_candidates: coincidences found in the injection-free frame.
    :type false_alarm_candidates: pandas.DataFrame | None
    :param false_alarm_candidates: foreground coincidences matching no injection.
    :type injections: pandas.DataFrame | None
    :param injections: the full injection table.
    :type livetime_days: float | None
    :param livetime_days: analysed livetime, days, used for rates.
    :type title: str
    :param title: report title.
    :type basename: str
    :param basename: file name stem for both outputs.
    :return: dict -- {"html": path, "markdown": path}.
    """
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    headline = _headline_numbers(recovery, coincidence_matched, candidates,
                                 background_candidates, false_alarm_candidates, livetime_days)
    by_class = _by_class_table(recovery)
    figures = [
        ("Detection efficiency by class", _efficiency_figure(recovery)),
        ("SNR recovery and timing", _recovery_figure(recovery)),
    ]
    if injections is not None:
        figures.insert(0, ("Injected population", _population_figure(injections)))

    html_path = os.path.join(outdir, f"{basename}.html")
    with open(html_path, "w") as fh:
        fh.write(_html(title, stamp, headline, by_class, trigger_summary, figures))

    md_path = os.path.join(outdir, f"{basename}.md")
    with open(md_path, "w") as fh:
        fh.write(_markdown(title, stamp, headline, by_class, trigger_summary, figures))

    return {"html": html_path, "markdown": md_path}


def _headline_numbers(recovery, coincidence_matched, candidates, background_candidates,
                      false_alarm_candidates, livetime_days):
    """Ordered (label, value) pairs summarising the run."""
    out = [("Injections", f"{len(recovery):,}"),
           ("Recovered per detector", f"{recovery['found'].mean() * 100:.1f}%")]
    found = recovery[recovery["found"]]
    if len(found) > 2:
        r = float(np.corrcoef(found["injected_snr"], found["recovered_snr"])[0, 1])
        out.append(("SNR correlation", f"{r:.3f}"))
        if "dt_s" in found:
            out.append(("Median timing offset", f"{found['dt_s'].median() * 1e3:+.1f} ms"))
    if coincidence_matched is not None and len(coincidence_matched):
        out.append(("Astrophysical in coincidence",
                    f"{coincidence_matched['found'].mean() * 100:.1f}%"))
    if candidates is not None:
        out.append(("Foreground coincidences", f"{len(candidates):,}"))
    if background_candidates is not None and livetime_days:
        out.append(("Background false-alarm rate",
                    f"{len(background_candidates) / livetime_days:.1f} / day"))
    if false_alarm_candidates is not None:
        out.append(("Coincidences with no injection", f"{len(false_alarm_candidates):,}"))
    return out


def _by_class_table(recovery):
    """Per-class injected, found, efficiency and median recovered SNR."""
    grouped = recovery.groupby(["category", "subclass"])
    table = grouped.agg(injected=("found", "size"), found=("found", "sum")).reset_index()
    table["efficiency"] = (table["found"] / table["injected"]).round(3)
    med = grouped["recovered_snr"].median().reset_index(name="median_recovered_snr")
    table = table.merge(med, on=["category", "subclass"])
    table["median_recovered_snr"] = table["median_recovered_snr"].round(2)
    return table.sort_values(["category", "efficiency"], ascending=[True, False])


def _population_figure(injections):
    """Injected classes and SNR distribution."""
    fig, axes = _figure(1, 2, (11, 3.6))
    order = injections.groupby("subclass").size().sort_values()
    colours = [C_BLUE if s in ("bbh", "bhns", "bns") else C_ORANGE for s in order.index]
    axes[0].barh(order.index, order.values, color=colours, height=0.7)
    axes[0].set_xlabel("injections")
    axes[0].set_title("By class", color=INK)
    axes[0].grid(axis="y", visible=False)
    for cat, colour in (("cbc", C_BLUE), ("glitch", C_ORANGE)):
        sel = injections[injections["category"] == cat]
        if len(sel):
            axes[1].hist(sel["network_snr"], bins=np.linspace(0, 100, 26), histtype="step",
                         lw=2, color=colour, label=cat)
    axes[1].set_xlabel("injected network SNR")
    axes[1].set_title("Signal-to-noise ratio", color=INK)
    axes[1].legend(fontsize=8)
    return _to_base64(fig)


def _efficiency_figure(recovery):
    """Efficiency against injected SNR, one panel per class."""
    classes = sorted(recovery["subclass"].unique())
    ncol = min(4, max(len(classes), 1))
    nrow = int(np.ceil(len(classes) / ncol))
    fig, axes = _figure(nrow, ncol, (3.2 * ncol, 2.6 * nrow), sharex=True, sharey=True)
    bins = np.linspace(4, 50, 12)
    for ax, name in zip(axes, classes):
        sel = recovery[recovery["subclass"] == name]
        curve = efficiency(sel, bins=bins, injected_snr_column="injected_snr")
        ax.step(curve["snr_mid"], curve["efficiency"], where="mid", lw=2, color=C_BLUE)
        ax.fill_between(curve["snr_mid"], 0, curve["efficiency"], step="mid",
                        alpha=0.12, color=C_BLUE)
        ax.set_title(f"{name} (n={len(sel)})", fontsize=9, color=INK)
        ax.set_ylim(0, 1.05)
    for ax in axes[len(classes):]:
        ax.set_visible(False)
    for ax in axes[max(len(classes) - ncol, 0):len(classes)]:
        ax.set_xlabel("injected SNR")
    for row in range(nrow):
        axes[row * ncol].set_ylabel("fraction recovered")
    return _to_base64(fig)


def _recovery_figure(recovery):
    """Recovered against injected SNR, and the timing residual."""
    found = recovery[recovery["found"]]
    fig, axes = _figure(1, 2, (11, 3.8))
    for cat, colour in (("cbc", C_BLUE), ("glitch", C_ORANGE)):
        sel = found[found["category"] == cat]
        if len(sel):
            axes[0].scatter(sel["injected_snr"], sel["recovered_snr"], s=10, alpha=0.5,
                            color=colour, edgecolors="none", label=cat)
    axes[0].set_xlabel("injected SNR")
    axes[0].set_ylabel("recovered EnWDF")
    axes[0].set_title("SNR recovery", color=INK)
    axes[0].legend(fontsize=8)
    if len(found):
        axes[1].hist(found["dt_s"] * 1e3, bins=40, color=C_BLUE, alpha=0.85)
    axes[1].set_xlabel("recovered $-$ injected time [ms]")
    axes[1].set_ylabel("injections")
    axes[1].set_title("Timing", color=INK)
    return _to_base64(fig)


def _figure(nrow, ncol, figsize, **kwargs):
    """A figure with flattened axes and the report's own styling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, **kwargs)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig, axes


def _to_base64(fig):
    """Render a figure to an inline PNG data URI and close it."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _html(title, stamp, headline, by_class, trigger_summary, figures):
    """Assemble the standalone HTML page."""
    tiles = "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div></div>' for label, value in headline)
    blocks = [f'<section><h2>Recovery by class</h2>{by_class.to_html(index=False, border=0)}</section>']
    if trigger_summary is not None:
        blocks.append('<section><h2>Triggers and clusters</h2>'
                      f'{trigger_summary.to_html(index=False, border=0)}</section>')
    for caption, uri in figures:
        blocks.append(f'<section><h2>{caption}</h2><img src="{uri}" alt="{caption}"></section>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ background: {SURFACE}; color: {INK}; margin: 0 auto; max-width: 1100px;
         padding: 2rem 1.25rem; font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; }}
  h2 {{ font-size: 1.05rem; margin: 2rem 0 .6rem; color: {INK}; }}
  .stamp {{ color: {INK_MUTED}; font-size: .85rem; margin-bottom: 1.5rem; }}
  .tiles {{ display: flex; flex-wrap: wrap; gap: .75rem; }}
  .tile {{ flex: 1 1 150px; border: 1px solid #e5e4e0; border-radius: 8px; padding: .7rem .9rem; }}
  .tile-label {{ color: {INK_MUTED}; font-size: .78rem; text-transform: uppercase;
                 letter-spacing: .04em; }}
  .tile-value {{ font-size: 1.3rem; font-weight: 600; margin-top: .2rem; }}
  section {{ overflow-x: auto; }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; font-size: .88rem; }}
  th, td {{ padding: .35rem .7rem; text-align: right; border-bottom: 1px solid #ecebe7; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  thead th {{ color: {INK_MUTED}; font-weight: 600; }}
</style></head>
<body>
<h1>{title}</h1>
<div class="stamp">generated {stamp}</div>
<div class="tiles">{tiles}</div>
{"".join(blocks)}
</body></html>
"""


def _markdown_table(frame):
    """Render a DataFrame as a Markdown pipe table."""
    header = [str(c) for c in frame.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in frame.itertuples(index=False)]
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
           "| " + " | ".join("-" * w for w in widths) + " |"]
    out += ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join(out)


def _markdown(title, stamp, headline, by_class, trigger_summary, figures=()):
    """Assemble the Markdown summary, with figures inlined as data URIs."""
    lines = [f"# {title}", "", f"*generated {stamp}*", "", "## Headline", ""]
    lines += [f"- **{label}**: {value}" for label, value in headline]
    lines += ["", "## Recovery by class", "", _markdown_table(by_class)]
    if trigger_summary is not None:
        lines += ["", "## Triggers and clusters", "", _markdown_table(trigger_summary)]
    for caption, uri in figures:
        lines += ["", f"## {caption}", "", f"![{caption}]({uri})"]
    return "\n".join(lines) + "\n"
