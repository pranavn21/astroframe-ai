import unittest

from astropy.table import Table
import numpy as np

from astroframe.metrics import background_statistics, median_star_fwhm


class MetricsTests(unittest.TestCase):
    def test_modern_and_legacy_centroid_names(self):
        y, x = np.indices((41, 41))
        image = 100 + 1000 * np.exp(-((x - 20)**2 + (y - 20)**2) / (2 * 1.2**2))
        for names in (("x_centroid", "y_centroid"), ("xcentroid", "ycentroid")):
            catalog = Table({names[0]: [20.], names[1]: [20.]})
            self.assertAlmostEqual(median_star_fwhm(image, catalog), 2.35482 * 1.2, places=3)

    def test_empty_catalog(self):
        self.assertIsNone(median_star_fwhm(np.zeros((8, 8)), None))

    def test_all_invalid_image(self):
        with self.assertRaisesRegex(ValueError, "no valid finite pixels"):
            background_statistics(np.full((8, 8), np.nan))
