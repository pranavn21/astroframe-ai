import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from analyze_session import main
from astroframe.plots import plot_background_timeline

class CliTests(unittest.TestCase):
    def test_session_csv_and_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "session"
            folder.mkdir()
            output = Path(directory) / "output"
            image = np.random.default_rng(123).normal(100, 2, (32, 32)).astype(np.float32)
            for name, timestamp in (("a.fit", "2026-01-10T20:02:00Z"), ("b.fits", "2026-01-10T20:01:00Z")):
                fits.writeto(folder / name, image, fits.Header({"DATE-OBS": timestamp, "EXPTIME": 10}))
            (folder / "corrupt.fit").write_text("invalid")
            with contextlib.redirect_stdout(io.StringIO()), self.assertLogs(level="WARNING"):
                code = main([str(folder), "--output-dir", str(output)])
            self.assertEqual(code, 0)
            frame = pd.read_csv(output / "session_metrics.csv")
            self.assertEqual(frame["filename"].tolist(), ["b.fits", "a.fit"])
            self.assertTrue(pd.to_datetime(frame["date_observed"]).is_monotonic_increasing)
            required = {"filename", "file_path", "date_observed", "median_fwhm_px", "star_count", "background_median",
                        "fwhm_delta_percent", "star_count_delta_percent", "background_delta_percent",
                        "high_fwhm", "low_star_count", "high_background", "is_anomaly"}
            self.assertTrue(required.issubset(frame.columns))
            for name in ("fwhm", "star_count", "background"):
                path = output / f"session_{name}_timeline.png"
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(plt.get_fignums(), [])

    def test_missing_timestamp_plot_and_empty_folder_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame = pd.DataFrame({"date_observed": [None, "bad"], "background_median": [100, 200]})
            plot_background_timeline(frame, folder / "missing.png")
            self.assertTrue((folder / "missing.png").is_file())
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    main([str(folder)])
            self.assertEqual(raised.exception.code, 1)
            self.assertIn("No FITS files found", error.getvalue())

    def test_session_with_guiding_log(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "session"
            folder.mkdir()
            header = fits.Header({"DATE-OBS": "2026-01-10T20:01:00Z", "EXPTIME": 10})
            image = np.random.default_rng(123).normal(100, 2, (32, 32)).astype(np.float32)
            fits.writeto(folder / "light.fit", image, header)
            (folder / "bad.fit").write_text("invalid")
            log = Path(directory) / "synthetic.txt"
            log.write_text("Guiding Begins at 2026-01-10 20:01:00\nFrame,Time,mount,RARawDistance,DECRawDistance\n1,0,Mount,3,4\n2,10,Mount,-3,-4\n")
            output = Path(directory) / "output"
            with contextlib.redirect_stdout(io.StringIO()), self.assertLogs(level="WARNING"):
                code = main([str(folder), "--output-dir", str(output), "--guide-log", str(log), "--guide-timezone", "UTC"])
            self.assertEqual(code, 0)
            frame = pd.read_csv(output / "session_metrics.csv")
            self.assertEqual(frame["guide_sample_count"].iloc[0], 2)
            self.assertEqual(frame["guide_total_rms_px"].iloc[0], 5)
            self.assertTrue(pd.isna(frame["guide_total_rms_arcsec"].iloc[0]))
            self.assertTrue((output / "session_guiding.csv").is_file())
