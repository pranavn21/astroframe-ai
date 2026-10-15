"""Analyze FITS frames from one session."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from astroframe.fits_reader import load_fits_image, read_fits_metadata
from astroframe.metrics import analyze_image

logger = logging.getLogger(__name__)


def find_fits_files(folder_path: str | Path) -> list[Path]:
    """Find FITS files recursively. For now, the folder should contain lights."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Session folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Session path is not a directory: {folder}")
    # TODO: distinguish lights from darks, flats, and bias frames.
    files = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".fit", ".fits", ".fts"}
    )
    if not files:
        raise ValueError(f"No FITS files found in: {folder}")
    return files


def analyze_session(folder_path: str | Path) -> pd.DataFrame:
    """Analyze each frame and sort by observation time, skipping failed files."""
    files = find_fits_files(folder_path)
    records, failures = [], []
    for number, path in enumerate(files, 1):
        print(f"[{number}/{len(files)}] Analyzing {path.name}")
        try:
            metadata = read_fits_metadata(path)
            metrics = analyze_image(load_fits_image(path))
            records.append({**metadata, **metrics, "file_path": str(path.resolve())})
        except Exception as exc:
            logger.warning("Failed to analyze %s: %s", path, exc)
            failures.append(str(path))
    if not records:
        raise ValueError(f"Failed to analyze all {len(files)} FITS files in: {folder_path}")

    result = pd.DataFrame(records)
    result["date_observed_raw"] = result["date_observed"]
    # Unqualified FITS dates are treated as UTC for the session timeline.
    result["date_observed"] = pd.to_datetime(
        result["date_observed"], format="mixed", errors="coerce", utc=True,
    )
    if result["date_observed"].isna().any():
        logger.warning("Some frames have missing or invalid timestamps; sorting them last")
    result = result.sort_values("date_observed", kind="stable", na_position="last")
    result = result.reset_index(drop=True)
    result = add_exposure_intervals(result)
    result.attrs["failed_files"] = failures
    return result


def add_exposure_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Use DATE-OBS as exposure start and EXPTIME as duration in seconds."""
    result = df.copy()
    dates = pd.to_datetime(result["date_observed"], format="mixed", errors="coerce", utc=True)
    seconds = pd.to_numeric(result["exposure_seconds"], errors="coerce")
    systems = result.get("time_system", pd.Series("UTC", index=result.index))
    starts, ends = [], []
    for date, duration, system in zip(dates, seconds, systems):
        start = end = pd.NaT
        known = pd.isna(system) or str(system).strip().upper() in ("", "UTC")
        if known and pd.notna(date) and pd.notna(duration) and np.isfinite(duration) and duration >= 0:
            try:
                start = date.as_unit("ns")
                end = start + pd.to_timedelta(duration, unit="s")
            except (ValueError, OverflowError):
                start = end = pd.NaT
        starts.append(start)
        ends.append(end)
    result["exposure_start"] = pd.Series(starts, index=result.index, dtype="datetime64[ns, UTC]")
    result["exposure_end"] = pd.Series(ends, index=result.index, dtype="datetime64[ns, UTC]")
    if result["exposure_end"].isna().any():
        logger.warning("Some frames have invalid exposure times or unsupported TIMESYS values")
    return result


def add_session_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add session medians and each frame's percentage deviations."""
    result = df.copy()
    for metric, baseline, deviation in (
        ("median_fwhm_px", "session_median_fwhm_px", "fwhm_delta_percent"),
        ("star_count", "session_median_star_count", "star_count_delta_percent"),
        ("background_median", "session_median_background", "background_delta_percent"),
    ):
        values = pd.to_numeric(
            result.get(metric, pd.Series(index=result.index, dtype=float)), errors="coerce",
        )
        values = values.where(np.isfinite(values))
        median = values.median() if values.notna().any() else np.nan
        result[baseline] = median
        result[deviation] = (
            (values - median) / median * 100 if pd.notna(median) and median != 0
            else np.nan
        )
    return result


def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Apply temporary thresholds; MAD or rolling baselines can improve these later."""
    result = df.copy()
    result["high_fwhm"] = (result["fwhm_delta_percent"] > 30).fillna(False)
    result["low_star_count"] = (result["star_count_delta_percent"] < -30).fillna(False)
    result["high_background"] = (result["background_delta_percent"] > 25).fillna(False)
    result["is_anomaly"] = result[["high_fwhm", "low_star_count", "high_background"]].any(axis=1)
    return result
