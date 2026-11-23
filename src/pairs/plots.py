"""Matplotlib figures.  Every figure carries a provenance stamp and its data is saved.

Stamp fields: data mode, universe, date range, formation/trading design, hedge method,
execution convention, cost assumptions and sample status (training / validation /
OOS / exploratory).  Data behind each figure is written to
``results/tables/figure_data/<figure>.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical order (dataviz reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f1f0ec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#eaf2fc", "#2a78d6", "#0d3c73"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.4, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


@dataclass(frozen=True)
class Stamp:
    data_mode: str
    universe: str
    dates: str
    design: str = "formation 12m / trading 6m"
    hedge: str = "static OLS"
    execution: str = "signal at close t, fill at next open (project-02 backtester)"
    costs: str = ""
    sample: str = ""

    def text(self) -> str:
        mode = ("SYNTHETIC DATA - not market evidence" if self.data_mode == "synthetic"
                else self.data_mode.upper())
        return (f"{mode} | universe: {self.universe} | {self.dates} | {self.design} | "
                f"hedge: {self.hedge} | {self.execution} | costs: {self.costs} | "
                f"sample: {self.sample}")

    def with_(self, **kw: str) -> Stamp:
        return replace(self, **kw)


def finish(fig: plt.Figure, path: str | Path, stamp: Stamp,
           data: pd.DataFrame | pd.Series | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.006, stamp.text(), fontsize=6, color=INK2, ha="left", va="bottom",
             wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    if data is not None:
        d = p.parent.parent / "tables" / "figure_data" / (p.stem + ".csv")
        d.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(d)
    return p


def pct(ax: plt.Axes, axis: str = "y", decimals: int = 0) -> None:
    f = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def heatmap(ax: plt.Axes, table: pd.DataFrame, fmt: str = "{:.2f}", center: float = 0.0,
            title: str = "") -> None:
    vals = table.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    lim = float(np.max(np.abs(finite - center))) if finite.size else 1.0
    lim = lim or 1.0
    im = ax.imshow(vals, cmap=DIVERGING, vmin=center - lim, vmax=center + lim, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index])
    ax.set_xlabel(str(table.columns.name or ""))
    ax.set_ylabel(str(table.index.name or ""))
    ax.grid(False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, "n/a" if not np.isfinite(v) else fmt.format(v), ha="center",
                    va="center", fontsize=8, color=INK)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0
