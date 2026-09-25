import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


def background_statistics(image: np.ndarray) -> dict:
    """
    Estimate the background level and noise of an astrophotography frame.

    Sigma clipping reduces the influence of stars and other bright objects.
    """

    finite_pixels = image[np.isfinite(image)]

    if finite_pixels.size == 0:
        raise ValueError("Image contains no valid finite pixels.")

    mean, median, std = sigma_clipped_stats(
        finite_pixels,
        sigma=3.0,
        maxiters=5,
    )

    return {
        "background_mean": float(mean),
        "background_median": float(median),
        "background_std": float(std),
    }

def detect_stars(
    image: np.ndarray,
    threshold_sigma: float = 5.0,
    fwhm_guess: float = 3.0,
):
    """
    Detect stars in an astrophotography frame using DAOStarFinder.
    """

    background = background_statistics(image)

    background_subtracted = (
        image - background["background_median"]
    )

    threshold = (
        threshold_sigma * background["background_std"]
    )

    finder = DAOStarFinder(
        fwhm=fwhm_guess,
        threshold=threshold,
    )

    sources = finder(background_subtracted)

    return sources

def _measure_star_fwhm(
    image: np.ndarray,
    x: float,
    y: float,
    background: float,
    radius: int = 5,
):
    """
    Estimate stellar FWHM using the second moments of a small
    background-subtracted stellar cutout.
    """

    x_center = int(round(x))
    y_center = int(round(y))

    y_min = y_center - radius
    y_max = y_center + radius + 1

    x_min = x_center - radius
    x_max = x_center + radius + 1

    if (
        y_min < 0
        or x_min < 0
        or y_max > image.shape[0]
        or x_max > image.shape[1]
    ):
        return None

    cutout = image[
        y_min:y_max,
        x_min:x_max,
    ].astype(np.float64)

    weights = cutout - background

    weights = np.clip(
        weights,
        a_min=0,
        a_max=None,
    )

    total = weights.sum()

    if total <= 0:
        return None

    yy, xx = np.indices(weights.shape)

    centroid_x = (xx * weights).sum() / total
    centroid_y = (yy * weights).sum() / total

    variance_x = (
        ((xx - centroid_x) ** 2 * weights).sum()
        / total
    )

    variance_y = (
        ((yy - centroid_y) ** 2 * weights).sum()
        / total
    )

    sigma = np.sqrt(
        (variance_x + variance_y) / 2
    )

    fwhm = 2.35482 * sigma

    if not np.isfinite(fwhm):
        return None

    return float(fwhm)

def median_star_fwhm(
    image: np.ndarray,
    sources,
) -> float | None:
    """
    Estimate the median FWHM across detected stars.
    """

    if sources is None or len(sources) == 0:
        return None

    background = background_statistics(image)

    measurements = []

    # To handle both versions of photutils, check for both possible column names
    if "x_centroid" in sources.colnames:
        x_column = "x_centroid"
        y_column = "y_centroid"
    elif "xcentroid" in sources.colnames:
        x_column = "xcentroid"
        y_column = "ycentroid"
    else:
        raise ValueError(
            "Star catalog does not contain recognized centroid columns. "
            f"Available columns: {sources.colnames}"
        )
    
    for source in sources:
        fwhm = _measure_star_fwhm(
            image=image,
            x=float(source[x_column]),
            y=float(source[y_column]),
            background=background["background_median"],
        )

        if fwhm is None:
            continue

        if 0.5 <= fwhm <= 20:
            measurements.append(fwhm)

    if not measurements:
        return None

    return float(np.median(measurements))

def analyze_image(image: np.ndarray) -> dict:
    """
    Calculate the initial AstroFrame quality metrics for one image.
    """

    background = background_statistics(image)

    sources = detect_stars(image)

    star_count = (
        len(sources)
        if sources is not None
        else 0
    )

    median_fwhm = median_star_fwhm(
        image,
        sources,
    )

    return {
        **background,
        "star_count": star_count,
        "median_fwhm_px": median_fwhm,
    }