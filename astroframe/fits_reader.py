from pathlib import Path

from astropy.io import fits


def read_fits_metadata(file_path: str) -> dict:
    """Read basic metadata from a FITS astrophotography frame."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")

    with fits.open(path) as hdul:
        header = hdul[0].header
        data = hdul[0].data

        metadata = {
            "filename": path.name,
            "width": data.shape[-1] if data is not None else None,
            "height": data.shape[-2] if data is not None else None,
            "exposure": header.get("EXPTIME"),
            "date_observed": header.get("DATE-OBS"),
            "filter": header.get("FILTER"),
            "object": header.get("OBJECT"),
            "gain": header.get("GAIN"),
            "temperature": header.get("CCD-TEMP"),
        }

    return metadata