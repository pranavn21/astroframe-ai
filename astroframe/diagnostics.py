"""Group unusual frames and describe the evidence behind them."""

import numpy as np
import pandas as pd


CAUSES = (
    "possible_guiding_related", "possible_transparency_related",
    "possible_focus_or_seeing_related", "possible_dither_contamination",
)
FLAGS = (
    "image_quality_degraded", "fwhm_degraded", "star_count_degraded",
    "background_increased", "background_decreased", "guiding_degraded",
    "dither_overlap", "missing_guiding_data",
)


def _values(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df.get(column, pd.Series(np.nan, index=df.index)), errors="coerce")
    return values.where(np.isfinite(values))


def _flag(df: pd.DataFrame, column: str) -> pd.Series:
    values = df.get(column, pd.Series(pd.NA, index=df.index))
    return values.map({True: True, False: False, "True": True, "False": False}).astype("boolean")


def _median(df: pd.DataFrame, column: str) -> float | None:
    values = _values(df, column).dropna()
    return float(values.median()) if len(values) else None


def _times(df: pd.DataFrame, column: str) -> pd.Series:
    values = df.get(column, pd.Series(pd.NaT, index=df.index))
    return pd.to_datetime(values, format="mixed", errors="coerce", utc=True)


def build_frame_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Use existing anomaly flags to identify possible causes for each frame."""
    result = df.copy()
    for source, target in (
        ("high_fwhm", "fwhm_degraded"), ("low_star_count", "star_count_degraded"),
        ("high_background", "background_increased"), ("low_background", "background_decreased"),
        ("high_guiding_rms", "guiding_degraded"), ("dither_during_exposure", "dither_overlap"),
    ):
        result[target] = _flag(result, source)
    image_flags = ["fwhm_degraded", "star_count_degraded", "background_increased", "background_decreased"]
    result["image_quality_degraded"] = result[image_flags].fillna(False).any(axis=1)
    rms_known = _values(result, "guide_total_rms_px").notna() | _values(result, "guide_total_rms_arcsec").notna()
    aligned = result.get("guide_alignment_status", pd.Series("aligned", index=result.index)).eq("aligned")
    result["missing_guiding_data"] = ~(
        rms_known & (_values(result, "guide_sample_count") > 0) & aligned & result["guiding_degraded"].notna()
    )
    guiding_normal = ~result["guiding_degraded"] & ~result["missing_guiding_data"]
    background_stable = (
        _values(result, "background_delta_percent").notna()
        & ~result["background_increased"] & ~result["background_decreased"]
    )
    result["possible_guiding_related"] = (
        result["fwhm_degraded"] & result["guiding_degraded"]
        & ~result["missing_guiding_data"] & background_stable & ~result["star_count_degraded"]
    ).fillna(False)
    result["possible_transparency_related"] = (
        (result["star_count_degraded"] | result["background_increased"] | result["background_decreased"])
        & guiding_normal
    ).fillna(False)
    result["possible_focus_or_seeing_related"] = (
        result["fwhm_degraded"] & guiding_normal & background_stable & ~result["star_count_degraded"]
    ).fillna(False)
    result["possible_dither_contamination"] = result["dither_overlap"].fillna(False)
    result["has_anomaly"] = (
        result["image_quality_degraded"] | result["guiding_degraded"].fillna(False)
        | result["dither_overlap"].fillna(False) | _flag(result, "is_anomaly").fillna(False)
    )
    result["insufficient_evidence"] = result["has_anomaly"] & ~result[list(CAUSES)].any(axis=1)
    return result


def classify_event_evidence(summary: dict) -> dict[str, bool]:
    """Keep cause labels as possibilities supported by the affected frames."""
    counts = summary["cause_counts"]
    result = {cause: counts[cause] > 0 for cause in CAUSES}
    result["insufficient_evidence"] = counts["insufficient_evidence"] > 0 or not any(result.values())
    return result


def summarize_event(df: pd.DataFrame) -> dict:
    """Summarize measurements without combining unrelated evidence across frames."""
    if df.empty:
        raise ValueError("Cannot summarize an empty event")
    frames = df if "has_anomaly" in df else build_frame_evidence(df)
    starts = _times(frames, "exposure_start").fillna(_times(frames, "date_observed"))
    ends = _times(frames, "exposure_end").fillna(starts)
    start, end = starts.min(), ends.max()
    image = {name: _median(frames, column) for name, column in (
        ("median_fwhm_px", "median_fwhm_px"), ("session_median_fwhm_px", "session_median_fwhm_px"),
        ("median_fwhm_delta_percent", "fwhm_delta_percent"), ("median_star_count", "star_count"),
        ("session_median_star_count", "session_median_star_count"),
        ("median_star_count_delta_percent", "star_count_delta_percent"),
        ("median_background", "background_median"), ("session_median_background", "session_median_background"),
        ("median_background_delta_percent", "background_delta_percent"),
    )}
    fwhm = _values(frames, "fwhm_delta_percent").dropna()
    image["max_fwhm_delta_percent"] = float(fwhm.max()) if len(fwhm) else None
    guiding = {"guide_sample_count_total": int(_values(frames, "guide_sample_count").fillna(0).clip(lower=0).sum())}
    for unit in ("px", "arcsec"):
        guiding[f"median_total_rms_{unit}"] = _median(frames, f"guide_total_rms_{unit}")
        guiding[f"session_median_total_rms_{unit}"] = _median(frames, f"session_median_guide_total_rms_{unit}")
        guiding[f"median_guiding_rms_delta_percent_{unit}"] = _median(frames, f"guiding_rms_delta_percent_{unit}")
        values = _values(frames, f"guide_total_rms_{unit}").dropna()
        guiding[f"max_total_rms_{unit}"] = float(values.max()) if len(values) else None
    flags = {}
    for name in FLAGS:
        values = frames[name]
        flags[name] = True if values.fillna(False).any() else None if values.isna().any() else False
    summary = {
        "event_start": start.isoformat() if pd.notna(start) else None,
        "event_end": end.isoformat() if pd.notna(end) else None,
        "duration_seconds": float((end - start).total_seconds()) if pd.notna(start) and pd.notna(end) else None,
        "frame_count": len(frames), "filenames": frames["filename"].astype(str).tolist(),
        "image_evidence": image, "guiding_evidence": guiding, "flags": flags,
        "cause_counts": {cause: int(frames[cause].sum()) for cause in (*CAUSES, "insufficient_evidence")},
    }
    summary["possible_causes"] = classify_event_evidence(summary)
    return summary


def _context(frames: pd.DataFrame) -> dict:
    return {
        "frame_count": len(frames), "median_fwhm_px": _median(frames, "median_fwhm_px"),
        "median_guide_total_rms_px": _median(frames, "guide_total_rms_px"),
        "median_guide_total_rms_arcsec": _median(frames, "guide_total_rms_arcsec"),
    }


def group_anomaly_events(df: pd.DataFrame, max_gap_seconds: float | None = None) -> list[dict]:
    """Group affected frames by start-time gaps, using 1.5 times cadence by default."""
    if max_gap_seconds is not None and (not np.isfinite(max_gap_seconds) or max_gap_seconds < 0):
        raise ValueError("Event gap must be finite and nonnegative")
    frames = build_frame_evidence(df)
    frames["_start"] = _times(frames, "exposure_start").fillna(_times(frames, "date_observed"))
    frames = frames.sort_values("_start", kind="stable", na_position="last").reset_index(drop=True)
    if max_gap_seconds is None:
        gaps = frames["_start"].dropna().diff().dt.total_seconds()
        gaps = gaps[gaps > 0]
        max_gap_seconds = float(gaps.median() * 1.5) if len(gaps) else 0
    groups, current = [], []
    for index in frames.index[frames["has_anomaly"]]:
        time = frames.loc[index, "_start"]
        previous = frames.loc[current[-1], "_start"] if current else pd.NaT
        close = pd.notna(time) and pd.notna(previous) and (time - previous).total_seconds() <= max_gap_seconds
        if current and not close:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    events = []
    for number, group in enumerate(groups, 1):
        event = summarize_event(frames.loc[group])
        event["event_id"] = number
        before, after = frames.iloc[:group[0]], frames.iloc[group[-1] + 1:]
        event["context"] = {
            "before": _context(before.loc[~before["has_anomaly"]].tail(2)),
            "after": _context(after.loc[~after["has_anomaly"]].head(2)),
        }
        events.append(event)
    return events


def calculate_session_trends(df: pd.DataFrame, threshold_percent: float = 20) -> dict:
    """Call a trend only when its fitted session-wide change exceeds 20 percent."""
    if not np.isfinite(threshold_percent) or threshold_percent < 0:
        raise ValueError("Trend threshold must be finite and nonnegative")
    times = _times(df, "exposure_start").fillna(_times(df, "date_observed"))
    trends = {}
    for name, column in (
        ("fwhm", "median_fwhm_px"), ("star_count", "star_count"), ("background", "background_median"),
        ("guiding_px", "guide_total_rms_px"), ("guiding_arcsec", "guide_total_rms_arcsec"),
    ):
        values = _values(df, column)
        valid = times.notna() & values.notna()
        result = {"trend": "insufficient_data", "change_percent": None}
        if valid.sum() >= 3:
            time = (times[valid] - times[valid].min()).dt.total_seconds()
            median = values[valid].median()
            if time.max() > 0 and median != 0:
                change = float(np.polyfit(time / time.max(), (values[valid] - median) / abs(median) * 100, 1)[0])
                result = {"trend": "increasing" if change > threshold_percent else "decreasing" if change < -threshold_percent else "stable",
                          "change_percent": change}
        trends[name] = result
    return trends
