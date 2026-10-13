import argparse
import logging
from pathlib import Path

import pandas as pd

from astroframe.plots import (
    plot_background_timeline, plot_fwhm_timeline, plot_star_count_timeline,
)
from astroframe.session import add_session_statistics, analyze_session, flag_anomalies


def _display(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if pd.notna(value) else "unavailable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a folder of FITS light frames.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="WARNING: %(message)s")
    print("AstroFrame Session Analysis\n---------------------------")

    try:
        frames = analyze_session(args.folder)
        frames = add_session_statistics(frames)
        frames = flag_anomalies(frames)
        name = args.folder.resolve().name or "session"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / f"{name}_metrics.csv"
        frames.to_csv(csv_path, index=False)
        plot_fwhm_timeline(frames, args.output_dir / f"{name}_fwhm_timeline.png")
        plot_star_count_timeline(frames, args.output_dir / f"{name}_star_count_timeline.png")
        plot_background_timeline(frames, args.output_dir / f"{name}_background_timeline.png")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print("\nSession Summary\n---------------------------")
    print(f"Frames analyzed: {len(frames)}")
    print(f"Frames failed: {len(frames.attrs['failed_files'])}")
    print(f"Median FWHM: {_display(frames['session_median_fwhm_px'].iloc[0])} px")
    print(f"Median stars: {_display(frames['session_median_star_count'].iloc[0], 0)}")
    print(f"Median background: {_display(frames['session_median_background'].iloc[0])}")
    anomalies = frames.loc[frames["is_anomaly"]]
    print(f"Potential anomalies: {len(anomalies)}")
    if not anomalies.empty:
        print(anomalies[["filename", "date_observed", "high_fwhm", "low_star_count", "high_background"]].to_string(index=False))
    print(f"\nSaved metrics: {csv_path}")
    print(f"Saved timeline PNGs in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
