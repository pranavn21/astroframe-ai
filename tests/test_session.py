import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from astropy.io import fits
import numpy as np
import pandas as pd

from astroframe.session import (
    add_session_statistics, analyze_session,
    find_fits_files, flag_anomalies,
)


class DiscoveryTests(unittest.TestCase):
    def test_recursive_extensions_and_deterministic_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "nested").mkdir()
            for name in ("b.FIT", "a.fits", "nested/c.FtS", "skip.fit.txt"):
                (folder / name).touch()
            (folder / "directory.fit").mkdir()
            expected = [folder / "a.fits", folder / "b.FIT", folder / "nested/c.FtS"]
            self.assertEqual(find_fits_files(folder), expected)
            self.assertEqual(find_fits_files(directory), expected)

    def test_invalid_paths_and_empty_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No FITS files found in"):
                find_fits_files(directory)
            path = Path(directory) / "file.txt"
            path.touch()
            with self.assertRaises(NotADirectoryError):
                find_fits_files(path)
            with self.assertRaises(FileNotFoundError):
                find_fits_files(Path(directory) / "missing")


class StatisticsTests(unittest.TestCase):
    def test_percentage_math_and_input_is_unchanged(self):
        original = pd.DataFrame({"median_fwhm_px": [2., 3., 4.],
                                 "star_count": [50, 100, 150],
                                 "background_median": [100, 200, 300]})
        result = add_session_statistics(original)
        np.testing.assert_allclose(result["fwhm_delta_percent"], [-100 / 3, 0, 100 / 3])
        np.testing.assert_allclose(result["star_count_delta_percent"], [-50, 0, 50])
        np.testing.assert_allclose(result["background_delta_percent"], [-50, 0, 50])
        self.assertEqual(result["session_median_fwhm_px"].tolist(), [3, 3, 3])
        self.assertNotIn("fwhm_delta_percent", original)

    def test_strict_thresholds_missing_values_and_median(self):
        frame = pd.DataFrame({"fwhm_delta_percent": [0, 30, 30.1, np.nan, 0, 0],
                              "star_count_delta_percent": [0, -30, 0, np.nan, -30.1, 0],
                              "background_delta_percent": [0, 25, 0, np.nan, 0, 25.1]})
        result = flag_anomalies(frame)
        self.assertEqual(result["is_anomaly"].tolist(), [False, False, True, False, True, True])
        self.assertEqual(result["high_fwhm"].tolist(), [False, False, True, False, False, False])

    def test_zero_all_missing_nonfinite_and_empty(self):
        for values in ([0, 0, 0], [np.nan, None], [np.inf, -np.inf], []):
            with self.subTest(values=values):
                frame = pd.DataFrame({metric: values for metric in (
                    "median_fwhm_px", "star_count", "background_median")})
                with np.errstate(divide="raise", invalid="raise"):
                    result = flag_anomalies(add_session_statistics(frame))
                self.assertTrue(result["fwhm_delta_percent"].isna().all())
                self.assertFalse(result["is_anomaly"].any())

    def test_partial_nan_uses_finite_median(self):
        frame = pd.DataFrame({"star_count": [100, None, 200, np.inf]})
        result = add_session_statistics(frame)
        self.assertEqual(result["session_median_star_count"].iloc[0], 150)
        self.assertTrue(pd.isna(result["star_count_delta_percent"].iloc[1]))
        self.assertTrue(pd.isna(result["star_count_delta_percent"].iloc[3]))

    def test_low_background_uses_a_strict_threshold(self):
        frame = pd.DataFrame({"fwhm_delta_percent": [0, 0], "star_count_delta_percent": [0, 0],
                              "background_delta_percent": [-25, -25.1]})
        result = flag_anomalies(frame)
        self.assertEqual(result["low_background"].tolist(), [False, True])
        self.assertEqual(result["is_anomaly"].tolist(), [False, True])


class SessionAnalysisTests(unittest.TestCase):
    def test_real_fits_io_sorting_raw_timestamps_and_corrupt_file(self):
        image = np.random.default_rng(123).normal(100, 2, (32, 32)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name, timestamp in (("a.fit", "2026-01-10T22:02:00"),
                                    ("b.FITS", "2026-01-10T22:01:00.000Z"),
                                    ("c.fts", "bad")):
                header = fits.Header({"DATE-OBS": timestamp, "EXPTIME": 10})
                fits.writeto(folder / name, image, header)
            (folder / "corrupt.fit").write_text("not a FITS image")
            with self.assertLogs("astroframe.session", level="WARNING"), contextlib.redirect_stdout(io.StringIO()):
                result = analyze_session(folder)
            self.assertEqual(result["filename"].tolist(), ["b.FITS", "a.fit", "c.fts"])
            self.assertEqual(result["date_observed_raw"].iloc[2], "bad")
            self.assertTrue(result["date_observed"].dropna().is_monotonic_increasing)
            self.assertEqual(result.index.tolist(), [0, 1, 2])
            self.assertEqual(len(result.attrs["failed_files"]), 1)
            self.assertEqual(result["width"].tolist(), [32] * 3)
            self.assertTrue(result["file_path"].map(lambda path: Path(path).is_absolute()).all())

    def test_all_frames_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "bad.fit").write_text("invalid")
            with self.assertLogs("astroframe.session", level="WARNING"), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Failed to analyze all 1"):
                    analyze_session(directory)
