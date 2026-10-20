import argparse
import json

from astroframe.fits_reader import (
    load_fits_image,
    read_fits_metadata,
)
from astroframe.metrics import analyze_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze one astrophotography FITS frame."
    )

    parser.add_argument(
        "file",
        help="Path to a FITS image.",
    )

    args = parser.parse_args(argv)

    try:
        metadata = read_fits_metadata(args.file)
        image = load_fits_image(args.file)
        metrics = analyze_image(image)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    result = {
        "metadata": metadata,
        "metrics": metrics,
    }

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
