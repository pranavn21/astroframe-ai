"""Create a small synthetic session for trying the CLI."""

import argparse
from pathlib import Path

from astropy.io import fits
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create three synthetic FITS frames and a guide log.")
    parser.add_argument("folder", type=Path, nargs="?", default=Path("sample_data/demo"),
                        help="New output folder (default: sample_data/demo).")
    args = parser.parse_args()
    if args.folder.exists():
        parser.error(f"Folder already exists: {args.folder}. Choose a new folder.")
    args.folder.mkdir(parents=True)
    rng = np.random.default_rng(123)
    y, x = np.indices((64, 64))
    lines = ["Guiding Begins at 2026-01-10 20:01:00", "Pixel scale = 1.50 arc-sec/px",
             "Frame,Time,mount,RARawDistance,DECRawDistance"]
    for number, width in enumerate((1.2, 1.2, 2.4)):
        image = rng.normal(100, 1, (64, 64)) + 1000 * np.exp(-((x - 32)**2 + (y - 32)**2) / (2 * width**2))
        header = fits.Header({"DATE-OBS": f"2026-01-10T20:0{number + 1}:00Z", "EXPTIME": 30,
                              "TIMESYS": "UTC", "OBJECT": "Synthetic stars", "ORIGIN": "AstroFrame demo"})
        fits.writeto(args.folder / f"light_{number}.fit", image.astype(np.float32), header)
        error = 3 if number == 2 else 1
        for step in range(3):
            lines.append(f"{number * 3 + step + 1},{number * 60 + step * 10},Mount,{error},{error / 2}")
    (args.folder / "synthetic_guide.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created synthetic FITS frames and guide log in {args.folder}")


if __name__ == "__main__":
    main()
