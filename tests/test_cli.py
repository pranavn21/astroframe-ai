import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests

from analyze_session import main
from astroframe.plots import plot_background_timeline
from config import REASONING_MODEL

class CliTests(unittest.TestCase):
    def make_anomalous_session(self, folder):
        folder.mkdir()
        for number in range(3):
            header = fits.Header({"DATE-OBS": f"2026-01-10T20:0{number}:00Z", "EXPTIME": 30})
            fits.writeto(folder / f"light_{number}.fit", np.zeros((8, 8)), header)
        return [{"background_mean": 100, "background_median": 100, "background_std": 2,
                 "star_count": 100, "median_fwhm_px": fwhm} for fwhm in (2, 2, 4)]

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

    @patch("requests.get")
    @patch("requests.post")
    def test_deterministic_events_do_not_call_ollama(self, post, get):
        with tempfile.TemporaryDirectory() as directory:
            folder, output = Path(directory) / "session", Path(directory) / "output"
            metrics = self.make_anomalous_session(folder)
            with patch("astroframe.session.analyze_image", side_effect=metrics), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(folder), "--output-dir", str(output)]), 0)
            events = json.loads((output / "session_events.json").read_text())["events"]
            self.assertEqual(len(events), 1)
            self.assertTrue(events[0]["possible_causes"]["insufficient_evidence"])
            self.assertFalse((output / "session_diagnoses.json").exists())
        post.assert_not_called()
        get.assert_not_called()

    @patch("requests.get", side_effect=requests.ConnectionError())
    @patch("requests.post")
    def test_offline_diagnosis_keeps_deterministic_outputs(self, post, get):
        with tempfile.TemporaryDirectory() as directory:
            folder, output = Path(directory) / "session", Path(directory) / "output"
            metrics = self.make_anomalous_session(folder)
            with patch("astroframe.session.analyze_image", side_effect=metrics), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertLogs(level="WARNING"):
                self.assertEqual(main([str(folder), "--output-dir", str(output), "--diagnose"]), 0)
            self.assertTrue((output / "session_metrics.csv").is_file())
            self.assertTrue((output / "session_events.json").is_file())
            results = json.loads((output / "session_diagnoses.json").read_text())["events"]
            self.assertIsNone(results[0]["diagnosis"])
            self.assertIn("Unable to reach Ollama", results[0]["error"])
        post.assert_not_called()

    @patch("requests.get")
    @patch("requests.post")
    def test_generated_fits_guiding_and_model_flow(self, post, get):
        diagnosis = {
            "classification": "possible_guiding_related", "confidence": "moderate",
            "summary": "Broader stars coincided with worse guiding; tracking is a possible cause.",
            "evidence": ["FWHM and guiding RMS increased while background stayed stable."],
            "alternatives": ["Wind or vibration may produce similar symptoms."],
            "recommendations": ["Review guiding during the affected exposure."],
        }
        get.return_value = Mock(json=Mock(return_value={"models": [{"name": REASONING_MODEL}]}))
        post.return_value = Mock(json=Mock(return_value={"message": {"content": json.dumps(diagnosis)}}))
        with tempfile.TemporaryDirectory() as directory:
            folder, output = Path(directory) / "session", Path(directory) / "output"
            folder.mkdir()
            y, x = np.indices((64, 64))
            rng = np.random.default_rng(123)
            lines = ["Guiding Begins at 2026-01-10 20:01:00", "Pixel scale = 1.50 arc-sec/px",
                     "Frame,Time,mount,RARawDistance,DECRawDistance"]
            for number, width in enumerate((1.2, 1.2, 2.4)):
                image = rng.normal(100, 1, (64, 64)) + 1000 * np.exp(-((x - 32)**2 + (y - 32)**2) / (2 * width**2))
                header = fits.Header({"DATE-OBS": f"2026-01-10T20:0{number + 1}:00Z", "EXPTIME": 30})
                fits.writeto(folder / f"light_{number}.fit", image.astype(np.float32), header)
                error = 3 if number == 2 else 1
                for step in range(3):
                    lines.append(f"{number * 3 + step + 1},{number * 60 + step * 10},Mount,{error},{error / 2}")
            log = Path(directory) / "synthetic.txt"
            log.write_text("\n".join(lines) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(folder), "--output-dir", str(output), "--guide-log", str(log),
                                       "--guide-timezone", "UTC", "--diagnose"]), 0)
            metrics = pd.read_csv(output / "session_metrics.csv")
            self.assertEqual(metrics["high_fwhm"].tolist(), [False, False, True])
            self.assertEqual(metrics["high_guiding_rms"].tolist(), [False, False, True])
            events = json.loads((output / "session_events.json").read_text())["events"]
            results = json.loads((output / "session_diagnoses.json").read_text())["events"]
            self.assertEqual(len(events), 1)
            self.assertTrue(events[0]["possible_causes"]["possible_guiding_related"])
            self.assertEqual(events[0]["filenames"], ["light_2.fit"])
            self.assertEqual(results[0]["event"], events[0])
            self.assertEqual(results[0]["diagnosis"], diagnosis)
