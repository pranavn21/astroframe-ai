"""Save session timelines as PNGs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _plot_timeline(df: pd.DataFrame, metric: str, title: str, ylabel: str,
                   output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    try:
        dates = pd.to_datetime(df["date_observed"], errors="coerce", utc=True)
        values = pd.to_numeric(df[metric], errors="coerce")
        valid = dates.notna() & values.notna()
        ax.plot(dates[valid], values[valid], marker="o")
        if valid.any():
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        else:
            ax.text(0.5, 0.5, "No valid timestamped measurements",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set(title=title, xlabel="Time (UTC)", ylabel=ylabel)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def plot_fwhm_timeline(df: pd.DataFrame, output_path: str | Path) -> None:
    _plot_timeline(df, "median_fwhm_px", "Median FWHM vs Time", "FWHM (pixels)", output_path)


def plot_star_count_timeline(df: pd.DataFrame, output_path: str | Path) -> None:
    _plot_timeline(df, "star_count", "Star Count vs Time", "Star count", output_path)


def plot_background_timeline(df: pd.DataFrame, output_path: str | Path) -> None:
    _plot_timeline(df, "background_median", "Median Background vs Time", "Background (image units)", output_path)
