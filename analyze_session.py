import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from astroframe.diagnostics import build_frame_evidence, calculate_session_trends, group_anomaly_events
from astroframe.guiding import correlate_guiding_to_frames, parse_phd2_log
from astroframe.plots import (
    plot_background_timeline, plot_fwhm_timeline, plot_star_count_timeline,
)
from astroframe.session import add_session_statistics, analyze_session, flag_anomalies


def _display(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if pd.notna(value) else "unavailable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a folder of FITS light frames.")
    parser.add_argument("folder", type=Path, help="Folder containing FITS light frames.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output folder (default: outputs).")
    parser.add_argument("--guide-log", type=Path, help="PHD2 guiding log.")
    parser.add_argument("--guide-timezone", help="Verified log timezone, such as UTC or America/Chicago.")
    parser.add_argument("--buffer-seconds", type=float, default=0, help="Extra seconds around each guiding window.")
    parser.add_argument("--event-gap-seconds", type=float, help="Maximum gap between affected exposure starts.")
    parser.add_argument("--diagnose", action="store_true", help="Explain events using the configured local Ollama model.")
    args = parser.parse_args(argv)
    if not args.guide_log and (args.guide_timezone or args.buffer_seconds != 0):
        parser.error("--guide-timezone and --buffer-seconds require --guide-log")
    logging.basicConfig(level=logging.WARNING, format="WARNING: %(message)s")
    print("AstroFrame Session Analysis\n---------------------------")

    try:
        frames = analyze_session(args.folder)
        frames = add_session_statistics(frames)
        frames = flag_anomalies(frames)
        guiding = None
        if args.guide_log:
            guiding = parse_phd2_log(args.guide_log, timezone=args.guide_timezone)
            frames = correlate_guiding_to_frames(frames, guiding, buffer_seconds=args.buffer_seconds)
        frames = build_frame_evidence(frames)
        events = group_anomaly_events(frames, max_gap_seconds=args.event_gap_seconds)
        trends = calculate_session_trends(frames)
        name = args.folder.resolve().name or "session"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / f"{name}_metrics.csv"
        frames.to_csv(csv_path, index=False)
        event_path = args.output_dir / f"{name}_events.json"
        event_path.write_text(json.dumps({"session": name, "trends": trends, "events": events}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if guiding is not None:
            guiding.to_csv(args.output_dir / f"{name}_guiding.csv", index=False)
        plot_fwhm_timeline(frames, args.output_dir / f"{name}_fwhm_timeline.png")
        plot_star_count_timeline(frames, args.output_dir / f"{name}_star_count_timeline.png")
        plot_background_timeline(frames, args.output_dir / f"{name}_background_timeline.png")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print("\nSession Summary\n---------------------------")
    print(f"Frames analyzed: {len(frames)}")
    print(f"Frames failed: {len(frames.attrs['failed_files'])}")
    print(f"Median FWHM: {_display(frames['session_median_fwhm_px'].iloc[0])} px")
    print(f"Median stars: {_display(frames['session_median_star_count'].iloc[0], 0)}")
    print(f"Median background: {_display(frames['session_median_background'].iloc[0])}")
    anomalies = frames.loc[frames["is_anomaly"]]
    print(f"Potential anomalies: {len(anomalies)}")
    print(f"Diagnostic events: {len(events)}")
    if guiding is not None:
        print(f"Frames with guiding samples: {int((frames['guide_sample_count'] > 0).sum())}")
    if not anomalies.empty:
        print(anomalies[["filename", "date_observed", "high_fwhm", "low_star_count", "high_background"]].to_string(index=False))
    print(f"\nSaved metrics: {csv_path}")
    print(f"Saved timeline PNGs in: {args.output_dir}")
    print(f"Saved event evidence: {event_path}")
    for event in events:
        causes = [name for name, supported in event["possible_causes"].items() if supported]
        print(f"Event {event['event_id']}: {event['event_start']} to {event['event_end']}; {event['frame_count']} frame(s)")
        print(f"FWHM delta: {_display(event['image_evidence']['median_fwhm_delta_percent'])}%")
        for unit in ("px", "arcsec"):
            value = event["guiding_evidence"][f"median_guiding_rms_delta_percent_{unit}"]
            if value is not None:
                print(f"Guiding RMS delta ({unit}): {_display(value)}%")
        print(f"Possible causes: {', '.join(causes)}")
    if args.diagnose:
        from astroframe.llm import diagnose_events

        path = args.output_dir / f"{name}_diagnoses.json"
        try:
            results = diagnose_events(events, output_path=path)
            for result in results:
                if result["diagnosis"] is not None:
                    diagnosis = result["diagnosis"]
                    print(f"Model interpretation, event {result['event']['event_id']}: {diagnosis['summary']}")
                    print(f"Confidence: {diagnosis['confidence']}")
                    for recommendation in diagnosis["recommendations"]:
                        print(f"- {recommendation}")
            print(f"Saved model diagnoses: {path}")
        except (OSError, RuntimeError, ValueError) as exc:
            logging.warning("Diagnosis step failed: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
