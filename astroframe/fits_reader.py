from pathlib import Path

import numpy as np
from astropy.io import fits


def _get_header_value(headers, *keys):
    """
    Return the first available FITS header value from a list
    of possible keyword names.
    """

    for header in headers:
        for key in keys:
            value = header.get(key)

            if value not in (None, ""):
                return value

    return None


def load_fits_image(file_path: str) -> np.ndarray:
    """
    Load the image data from a FITS file as a 2D NumPy array.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")

    with fits.open(path, memmap=False) as hdul:
        image_hdu = next(
            (hdu for hdu in hdul if hdu.data is not None),
            None,
        )

        if image_hdu is None:
            raise ValueError(
                f"No image data found in FITS file: {file_path}"
            )

        data = np.asarray(image_hdu.data, dtype=np.float32)

    data = np.squeeze(data)

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D FITS image, but received shape {data.shape}"
        )

    return data


def read_fits_metadata(file_path: str) -> dict:
    """
    Extract useful astrophotography metadata from a FITS frame.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")

    with fits.open(path, memmap=False) as hdul:
        primary_header = hdul[0].header

        image_hdu = next(
            (hdu for hdu in hdul if hdu.data is not None),
            None,
        )

        if image_hdu is None:
            raise ValueError(
                f"No image data found in FITS file: {file_path}"
            )

        image_header = image_hdu.header
        data = np.squeeze(image_hdu.data)

        headers = [image_header, primary_header]

        metadata = {
            "filename": path.name,
            "width": int(data.shape[-1]),
            "height": int(data.shape[-2]),
            "exposure_seconds": _get_header_value(
                headers,
                "EXPTIME",
                "EXPOSURE",
            ),
            "date_observed": _get_header_value(
                headers,
                "DATE-OBS",
                "DATEOBS",
            ),
            "filter": _get_header_value(
                headers,
                "FILTER",
            ),
            "object": _get_header_value(
                headers,
                "OBJECT",
                "OBJNAME",
            ),
            "gain": _get_header_value(
                headers,
                "GAIN",
                "EGAIN",
            ),
            "temperature_c": _get_header_value(
                headers,
                "CCD-TEMP",
                "CCD_TEMP",
                "CCDTEMP",
                "SENSOR_TEMP",
            ),
            "camera": _get_header_value(
                headers,
                "INSTRUME",
                "CAMERA",
            ),
        }

    return metadata