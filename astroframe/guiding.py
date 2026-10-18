"""Read PHD2 guide logs and match samples to FITS exposures."""

import csv
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

from astroframe.session import add_exposure_intervals

logger = logging.getLogger(__name__)


def _number(value: object) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _parse_time(value: object, timezone: str | None = None) -> pd.Timestamp:
    time = pd.to_datetime(value, errors="coerce")
    if pd.isna(time):
        return pd.NaT
    if time.tzinfo is None and timezone:
        time = time.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    if pd.notna(time) and time.tzinfo is not None:
        time = time.tz_convert("UTC")
    try:
        return time.as_unit("ns") if pd.notna(time) else pd.NaT
    except (ValueError, OverflowError):
        return pd.NaT


def parse_phd2_log(file_path: str | Path, *, timezone: str | None = None) -> pd.DataFrame:
    """Read guiding sections; unqualified dates keep their unknown timezone."""
    if timezone:
        pd.Timestamp("2000-01-01").tz_localize(timezone)
    records = []
    header = []
    start = pd.NaT
    scale = np.nan
    session = 0
    active = paused = False

    with Path(file_path).open(encoding="utf-8-sig", errors="replace") as log:
        for line_number, line in enumerate(log, 1):
            line = line.strip()
            if line.startswith("Guiding Begins at "):
                start = _parse_time(line.removeprefix("Guiding Begins at "), timezone)
                session += 1
                active, paused, header, scale = True, False, [], np.nan
                continue
            if line.startswith(("Guiding Ends", "Calibration Begins")):
                active, header = False, []
                continue
            if active and "Pixel scale =" in line:
                match = re.search(r"Pixel scale\s*=\s*([\d.eE+-]+)\s*arc[- ]?sec/px", line)
                scale = _number(match.group(1)) if match else np.nan
                scale = scale if scale > 0 else np.nan
                continue
            if active and line.startswith("INFO:"):
                text = line.lower()
                event = "info"
                if "dither by" in text:
                    event = "dither"
                elif "resum" in text or "unpaus" in text:
                    event, paused = "resume", False
                elif "paus" in text:
                    event, paused = "pause", True
                match = re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?", line)
                records.append({
                    "timestamp": _parse_time(match.group(), timezone) if match else pd.NaT,
                    "event_type": event, "guiding_session_id": session,
                    "line_number": line_number, "message": line,
                })
                continue
            if not line:
                continue
            try:
                fields = next(csv.reader([line], strict=True))
                names = [field.strip().lower() for field in fields]
                if names[0] == "frame" and ("time" in names or "timestamp" in names):
                    header = names
                    if session == 0:
                        session, active = 1, True
                    continue
                if not active or not header:
                    continue
                if len(fields) < len(header):
                    raise ValueError("row has fewer fields than its header")
                row = dict(zip(header, fields))
                raw_time = row.get("timestamp", row.get("time"))
                seconds = _number(raw_time)
                if pd.notna(seconds):
                    if seconds < 0:
                        raise ValueError("negative elapsed time")
                    time = start + pd.to_timedelta(seconds, unit="s") if pd.notna(start) else pd.NaT
                else:
                    time = _parse_time(raw_time, timezone)
                    if pd.isna(time):
                        raise ValueError("invalid guide timestamp")
                device = row.get("mount", "Mount").upper()
                event = "guiding"
                if device == "DROP" or _number(row.get("errorcode", 0)) != 0:
                    event = "dropped"
                elif device == "AO":
                    event = "ao"
                elif paused:
                    event = "paused"
                record = {
                    "timestamp": time, "event_type": event, "guiding_session_id": session,
                    "ra_error_px": _number(row.get("rarawdistance", row.get("ra_error_px"))),
                    "dec_error_px": _number(row.get("decrawdistance", row.get("dec_error_px"))),
                    "ra_error_arcsec": _number(row.get("ra_error_arcsec")),
                    "dec_error_arcsec": _number(row.get("dec_error_arcsec")),
                    "ra_pulse_ms": _number(row.get("raduration", row.get("ra_pulse_ms"))),
                    "dec_pulse_ms": _number(row.get("decduration", row.get("dec_pulse_ms"))),
                    "snr": _number(row.get("snr")), "star_mass": _number(row.get("starmass", row.get("star_mass"))),
                    "line_number": line_number,
                }
                for axis in ("ra", "dec"):
                    if pd.isna(record[f"{axis}_error_arcsec"]) and pd.notna(scale):
                        record[f"{axis}_error_arcsec"] = record[f"{axis}_error_px"] * scale
                if event == "guiding" and not any(
                    pd.notna(record[f"ra_error_{unit}"]) and pd.notna(record[f"dec_error_{unit}"])
                    for unit in ("px", "arcsec")
                ):
                    raise ValueError("missing RA/DEC measurements")
                records.append(record)
            except (csv.Error, ValueError, TypeError, OverflowError) as exc:
                logger.warning("Skipping guide log line %d: %s", line_number, exc)

    result = pd.DataFrame(records).reindex(columns=[
        "timestamp", "ra_error_px", "dec_error_px", "ra_error_arcsec", "dec_error_arcsec",
        "ra_pulse_ms", "dec_pulse_ms", "snr", "star_mass", "event_type",
        "guiding_session_id", "line_number", "message",
    ])
    if result.empty or not result["event_type"].eq("guiding").any():
        raise ValueError(f"No valid guiding samples found in: {file_path}")
    for unit in ("px", "arcsec"):
        result[f"total_error_{unit}"] = np.hypot(result[f"ra_error_{unit}"], result[f"dec_error_{unit}"])
    if any(pd.notna(time) and time.tzinfo is None for time in result["timestamp"]):
        logger.warning("Guide log timezone is unknown; provide a verified timezone for exposure matching")
    if result.loc[result["event_type"].eq("dither"), "timestamp"].isna().any():
        logger.warning("Some dither events have no timestamps; their exposure overlap is unknown")
    return result


def _guide_samples(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    if "event_type" in df:
        df = df.loc[df["event_type"].eq("guiding")]
    values = df.reindex(columns=[f"ra_error_{unit}", f"dec_error_{unit}"])
    values = values.apply(pd.to_numeric, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_guiding_rms(df: pd.DataFrame) -> dict[str, float]:
    """Calculate RMS from paired, finite RA/DEC samples in each available unit."""
    result = {}
    for unit in ("px", "arcsec"):
        samples = _guide_samples(df, unit)
        ra, dec = np.sqrt((samples ** 2).mean()).tolist()
        result.update({f"ra_rms_{unit}": ra, f"dec_rms_{unit}": dec,
                       f"total_rms_{unit}": float(np.hypot(ra, dec))})
    return result


def add_guiding_statistics(df: pd.DataFrame, threshold_percent: float = 50) -> pd.DataFrame:
    """Flag guiding RMS above a temporary session-relative threshold."""
    if not np.isfinite(threshold_percent) or threshold_percent < 0:
        raise ValueError("Guiding threshold must be finite and nonnegative")
    result = df.copy()
    counts = pd.to_numeric(result["guide_sample_count"], errors="coerce")
    for unit in ("px", "arcsec"):
        column = f"guide_total_rms_{unit}"
        values = pd.to_numeric(result[column], errors="coerce")
        values = values.where(np.isfinite(values) & (counts > 0) & (values >= 0))
        median = values.median() if values.notna().any() else np.nan
        result[f"session_median_guide_total_rms_{unit}"] = median
        delta = (values - median) / median * 100 if pd.notna(median) and median > 0 else values * np.nan
        result[f"guiding_rms_delta_percent_{unit}"] = delta
        result[f"high_guiding_rms_{unit}"] = (delta > threshold_percent).astype("boolean").mask(delta.isna())
    arcsec = pd.to_numeric(result["guide_total_rms_arcsec"], errors="coerce").notna()
    result["guiding_rms_delta_percent"] = result["guiding_rms_delta_percent_arcsec"].where(
        arcsec, result["guiding_rms_delta_percent_px"],
    )
    result["high_guiding_rms"] = result["high_guiding_rms_arcsec"].where(arcsec, result["high_guiding_rms_px"])
    return result


def correlate_guiding_to_frames(
    frames_df: pd.DataFrame, guiding_df: pd.DataFrame, *, buffer_seconds: float = 0,
) -> pd.DataFrame:
    """Add guiding RMS and peaks for each inclusive exposure window."""
    if not np.isfinite(buffer_seconds) or buffer_seconds < 0:
        raise ValueError("buffer_seconds must be finite and nonnegative")
    buffer = pd.to_timedelta(buffer_seconds, unit="s")
    frames = add_exposure_intervals(frames_df)
    guide = guiding_df.reset_index(drop=True).copy()
    times = []
    for value in guide["timestamp"]:
        time = _parse_time(value)
        times.append(time if pd.notna(time) and time.tzinfo is not None else pd.NaT)
    times = pd.Series(times, dtype="datetime64[ns, UTC]")
    measured = guide["event_type"].eq("guiding")
    missing = measured & times.isna()
    status = "aligned"
    if missing.any():
        status = "partial_time_basis" if (measured & times.notna()).any() else "unknown_time_basis"
        logger.warning("Some guide samples have missing or unknown timezones; they will not be matched")
    dithers = guide["event_type"].eq("dither")
    records = []
    for _, frame in frames.iterrows():
        start, end = frame["exposure_start"], frame["exposure_end"]
        valid = pd.notna(start) and pd.notna(end)
        try:
            mask = (times >= start - buffer) & (times <= end + buffer) if valid else pd.Series(False, index=guide.index)
        except (ValueError, OverflowError):
            valid = False
            mask = pd.Series(False, index=guide.index)
        window = guide.loc[mask]
        samples = {unit: _guide_samples(window, unit) for unit in ("px", "arcsec")}
        record = {"guide_sample_count": len(samples["px"].index.union(samples["arcsec"].index)),
                  "guide_alignment_status": status if valid else "invalid_exposure"}
        record.update({f"guide_{key}": value for key, value in calculate_guiding_rms(window).items()})
        for unit, values in samples.items():
            record[f"guide_ra_peak_abs_{unit}"] = values.iloc[:, 0].abs().max()
            record[f"guide_dec_peak_abs_{unit}"] = values.iloc[:, 1].abs().max()
            record[f"guide_total_peak_{unit}"] = np.hypot(values.iloc[:, 0], values.iloc[:, 1]).max()
        overlap = valid and (dithers & (times >= start) & (times <= end)).any()
        record["dither_during_exposure"] = True if overlap else pd.NA if (dithers & times.isna()).any() or not valid else False
        records.append(record)
    summary = pd.DataFrame(records, index=frames.index)
    for column in summary:
        frames[column] = summary[column]
    return add_guiding_statistics(frames) if records else frames
