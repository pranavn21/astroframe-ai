from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from astroframe.guiding import add_guiding_statistics, calculate_guiding_rms, correlate_guiding_to_frames, parse_phd2_log
from astroframe.session import add_exposure_intervals


LOG = """Calibration Begins at 2026-01-10 20:00:00
Direction,Step,dx,dy,x,y,Dist
West,1,0,0,0,0,0
Guiding Begins at 2026-01-10 20:01:00
Pixel scale = 1.50 arc-sec/px
Frame,Time,mount,RARawDistance,DECRawDistance,RADuration,DECDuration,SNR,StarMass,ErrorCode
1,0,Mount,3,4,100,200,20,1000,0
2,2,Mount,-3,-4,0,0,18,900,0
3,3,DROP,,,0,0,1,100,1
4,broken,Mount,1,1,0,0,20,1000,0
5,4,Mount,99,99,0,0,20,1000,1
6,5,AO,99,99,0,0,20,1000,0
INFO: Guiding paused
7,6,Mount,99,99,0,0,20,1000,0
INFO: Guiding resumed
8,7,Mount,3,4,0,0,20,1000,0
Guiding Ends at 2026-01-10 20:01:10
Guiding Begins at 2026-01-10 20:02:00
Frame,Time,mount,RARawDistance,DECRawDistance
1,0,Mount,0.5,1
"""


class ParserTests(unittest.TestCase):
    def parse_log(self, text=LOG, timezone="UTC"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.txt"
            path.write_text(text)
            return parse_phd2_log(path, timezone=timezone)

    def test_sessions_units_and_bad_rows(self):
        with self.assertLogs("astroframe.guiding", level="WARNING"):
            guide = self.parse_log()
        samples = guide.loc[guide["event_type"].eq("guiding")]
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples["guiding_session_id"].tolist(), [1, 1, 1, 2])
        first = samples.iloc[0]
        self.assertEqual(first["timestamp"], pd.Timestamp("2026-01-10T20:01:00Z"))
        self.assertEqual(first["total_error_px"], 5)
        self.assertEqual(first["total_error_arcsec"], 7.5)
        self.assertEqual(first["ra_pulse_ms"], 100)
        self.assertEqual(first["dec_pulse_ms"], 200)
        self.assertEqual(first["snr"], 20)
        self.assertEqual(first["star_mass"], 1000)
        self.assertTrue(pd.isna(samples.iloc[-1]["total_error_arcsec"]))
        self.assertEqual(calculate_guiding_rms(guide.iloc[:-1])["total_rms_px"], 5)
        for event in ("dropped", "ao", "paused", "pause", "resume"):
            self.assertIn(event, guide["event_type"].tolist())

    def test_unknown_timezone_is_not_assumed_to_be_utc(self):
        with self.assertLogs("astroframe.guiding", level="WARNING"):
            guide = self.parse_log(timezone=None)
            frames = pd.DataFrame({"date_observed": ["2026-01-10T20:01:00Z"], "exposure_seconds": [10]})
            result = correlate_guiding_to_frames(frames, guide)
        self.assertIsNone(guide["timestamp"].dropna().iloc[0].tzinfo)
        self.assertEqual(result["guide_sample_count"].iloc[0], 0)
        self.assertEqual(result["guide_alignment_status"].iloc[0], "unknown_time_basis")
        self.assertTrue(pd.isna(result["guide_total_rms_px"].iloc[0]))

    def test_local_timezone_and_explicit_offsets(self):
        with self.assertLogs("astroframe.guiding", level="WARNING"):
            guide = self.parse_log(timezone="America/Chicago")
        self.assertEqual(guide["timestamp"].iloc[0], pd.Timestamp("2026-01-11T02:01:00Z"))
        text = "Frame,Timestamp,ra_error_arcsec,dec_error_arcsec\n1,2026-01-10T20:01:00-06:00,0.3,-0.4\n"
        guide = self.parse_log(text, timezone=None)
        self.assertEqual(guide["timestamp"].iloc[0], pd.Timestamp("2026-01-11T02:01:00Z"))
        self.assertEqual(guide["total_error_arcsec"].iloc[0], 0.5)
        self.assertTrue(pd.isna(guide["total_error_px"].iloc[0]))

    def test_dither_timestamps_and_untimed_markers(self):
        text = """Guiding Begins at 2026-01-10 20:01:00
Frame,Time,RARawDistance,DECRawDistance
1,0,3,4
INFO: DITHER by 1, 1
INFO: 2026-01-10 20:01:02 DITHER by 1, 1
"""
        with self.assertLogs("astroframe.guiding", level="WARNING"):
            guide = self.parse_log(text)
        events = guide.loc[guide["event_type"].eq("dither")]
        self.assertTrue(pd.isna(events["timestamp"].iloc[0]))
        self.assertEqual(events["timestamp"].iloc[1], pd.Timestamp("2026-01-10T20:01:02Z"))

    def test_missing_start_and_ambiguous_dst_remain_unavailable(self):
        text = "Frame,Time,RARawDistance,DECRawDistance\n1,2,3,4\n"
        guide = self.parse_log(text)
        self.assertTrue(guide["timestamp"].isna().all())
        text = "Guiding Begins at 2026-11-01 01:30:00\n" + text
        guide = self.parse_log(text, timezone="America/Chicago")
        self.assertTrue(guide["timestamp"].isna().all())

    def test_empty_invalid_and_missing_logs(self):
        with self.assertRaisesRegex(ValueError, "No valid guiding samples"):
            self.parse_log("Calibration Begins at 2026-01-10 20:00:00\n")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                parse_phd2_log(Path(directory) / "missing.txt")
        text = "Guiding Begins at 2026-01-10 20:01:00Z\nFrame,Time,RARawDistance,DECRawDistance\n1,0,3,4\n2,1,NaN,4\n3,2,inf,4\n4,3,1\n"
        with self.assertLogs("astroframe.guiding", level="WARNING") as warnings:
            guide = self.parse_log(text)
        self.assertEqual(len(guide), 1)
        self.assertEqual(len(warnings.output), 3)


class GuidingTests(unittest.TestCase):
    def setUp(self):
        self.frames = pd.DataFrame({"date_observed": ["2026-01-10T20:01:00Z", "2026-01-10T21:00:00Z", None],
                                    "exposure_seconds": [10, 10, 10]})
        self.guide = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-01-10T20:00:59Z", "2026-01-10T20:01:00Z",
                                         "2026-01-10T20:01:10Z", "2026-01-10T20:01:11Z"]),
            "ra_error_px": [30, 3, -3, 30], "dec_error_px": [40, 4, -4, 40],
            "event_type": ["guiding"] * 4,
        })

    def test_rms_and_missing_pairs(self):
        samples = pd.DataFrame({"ra_error_px": [3, 3, None, np.inf], "dec_error_px": [4, 4, 1, 2]})
        rms = calculate_guiding_rms(samples)
        self.assertEqual(rms["ra_rms_px"], 3)
        self.assertEqual(rms["dec_rms_px"], 4)
        self.assertEqual(rms["total_rms_px"], 5)
        self.assertTrue(pd.isna(rms["total_rms_arcsec"]))
        self.assertTrue(all(pd.isna(value) for value in calculate_guiding_rms(pd.DataFrame()).values()))

    def test_guiding_baselines_keep_units_separate(self):
        frames = pd.DataFrame({"guide_sample_count": [10, 10, 10, 0],
                               "guide_total_rms_px": [1, 1, 3, np.nan],
                               "guide_total_rms_arcsec": [2, 2, 6, np.nan]})
        result = add_guiding_statistics(frames)
        self.assertEqual(result["session_median_guide_total_rms_px"].iloc[0], 1)
        self.assertEqual(result["session_median_guide_total_rms_arcsec"].iloc[0], 2)
        self.assertEqual(result["guiding_rms_delta_percent"].iloc[2], 200)
        self.assertEqual(result["high_guiding_rms"].iloc[:3].tolist(), [False, False, True])
        self.assertTrue(pd.isna(result["high_guiding_rms"].iloc[3]))

    def test_zero_and_missing_guiding_baselines_stay_unknown(self):
        for values in ([0, 0], [np.nan, np.nan]):
            frames = pd.DataFrame({"guide_sample_count": [2, 2],
                                   "guide_total_rms_px": values, "guide_total_rms_arcsec": [np.nan, np.nan]})
            with np.errstate(divide="raise", invalid="raise"):
                result = add_guiding_statistics(frames)
            self.assertTrue(result["guiding_rms_delta_percent"].isna().all())
            self.assertTrue(result["high_guiding_rms"].isna().all())

    def test_exposure_windows_and_no_overlap(self):
        self.frames.attrs["failed_files"] = ["bad.fit"]
        with self.assertLogs("astroframe.session", level="WARNING"):
            result = correlate_guiding_to_frames(self.frames, self.guide)
        self.assertEqual(result["guide_sample_count"].tolist(), [2, 0, 0])
        self.assertEqual(result["guide_total_rms_px"].iloc[0], 5)
        self.assertEqual(result["guide_ra_peak_abs_px"].iloc[0], 3)
        self.assertEqual(result["guide_dec_peak_abs_px"].iloc[0], 4)
        self.assertEqual(result["guide_total_peak_px"].iloc[0], 5)
        self.assertTrue(result["guide_total_rms_px"].iloc[1:].isna().all())
        self.assertEqual(result.attrs["failed_files"], ["bad.fit"])

    def test_buffer_and_dither_use_different_bounds(self):
        event = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-10T20:01:11Z")], "event_type": ["dither"]})
        guide = pd.concat([self.guide, event], ignore_index=True)
        result = correlate_guiding_to_frames(self.frames.iloc[:1], guide, buffer_seconds=1)
        self.assertEqual(result["guide_sample_count"].iloc[0], 4)
        self.assertFalse(result["dither_during_exposure"].iloc[0])
        guide.loc[4, "timestamp"] = pd.Timestamp("2026-01-10T20:01:05Z")
        result = correlate_guiding_to_frames(self.frames.iloc[:1], guide)
        self.assertTrue(result["dither_during_exposure"].iloc[0])
        guide.loc[4, "timestamp"] = pd.NaT
        result = correlate_guiding_to_frames(self.frames.iloc[:1], guide)
        self.assertTrue(pd.isna(result["dither_during_exposure"].iloc[0]))
        for buffer in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                correlate_guiding_to_frames(self.frames, guide, buffer_seconds=buffer)

    def test_exposure_intervals_and_bad_headers(self):
        frames = pd.DataFrame({"date_observed": ["2026-01-10T20:01:00Z"] * 5,
                               "exposure_seconds": [10, -1, np.inf, 10, 1e100],
                               "time_system": [None, None, None, "TT", None]})
        with self.assertLogs("astroframe.session", level="WARNING"):
            result = add_exposure_intervals(frames)
        self.assertEqual(result["exposure_end"].iloc[0], pd.Timestamp("2026-01-10T20:01:10Z"))
        self.assertTrue(result["exposure_end"].iloc[1:].isna().all())
