import json
import unittest

import numpy as np
import pandas as pd

from astroframe.diagnostics import build_frame_evidence, calculate_session_trends, group_anomaly_events, summarize_event


def frame(**changes):
    row = {
        "filename": "light.fit", "date_observed": "2026-01-10T00:00:00Z",
        "exposure_start": "2026-01-10T00:00:00Z", "exposure_end": "2026-01-10T00:05:00Z",
        "median_fwhm_px": 3, "session_median_fwhm_px": 2, "fwhm_delta_percent": 50,
        "star_count": 100, "session_median_star_count": 100, "star_count_delta_percent": 0,
        "background_median": 100, "session_median_background": 100, "background_delta_percent": 0,
        "high_fwhm": True, "low_star_count": False, "high_background": False, "low_background": False,
        "is_anomaly": True, "guide_sample_count": 10, "guide_total_rms_px": 2,
        "guide_total_rms_arcsec": np.nan, "session_median_guide_total_rms_px": 1,
        "guiding_rms_delta_percent_px": 100, "high_guiding_rms": True,
        "guide_alignment_status": "aligned", "dither_during_exposure": False,
    }
    row.update(changes)
    return row


class EvidenceTests(unittest.TestCase):
    def test_guiding_transparency_and_focus_patterns(self):
        cases = [
            (frame(), "possible_guiding_related"),
            (frame(low_star_count=True, high_background=True, high_guiding_rms=False,
                   star_count_delta_percent=-50, background_delta_percent=50), "possible_transparency_related"),
            (frame(high_guiding_rms=False), "possible_focus_or_seeing_related"),
            (frame(dither_during_exposure=True), "possible_dither_contamination"),
        ]
        for row, cause in cases:
            with self.subTest(cause=cause):
                result = build_frame_evidence(pd.DataFrame([row])).iloc[0]
                self.assertTrue(result[cause])
        result = build_frame_evidence(pd.DataFrame([frame()])).iloc[0]
        self.assertFalse(result["possible_transparency_related"])
        self.assertFalse(result["possible_focus_or_seeing_related"])

    def test_missing_guiding_is_not_normal_guiding(self):
        row = frame(guide_sample_count=0, high_guiding_rms=pd.NA, guide_total_rms_px=np.nan)
        result = build_frame_evidence(pd.DataFrame([row])).iloc[0]
        self.assertTrue(result["missing_guiding_data"])
        self.assertTrue(result["insufficient_evidence"])
        self.assertFalse(result["possible_focus_or_seeing_related"])
        self.assertFalse(result["possible_guiding_related"])

    def test_conflicting_image_changes_remain_unclear(self):
        for row in (frame(low_star_count=True, star_count_delta_percent=-60),
                    frame(high_background=True, background_delta_percent=60)):
            result = build_frame_evidence(pd.DataFrame([row])).iloc[0]
            self.assertFalse(result["possible_guiding_related"])
            self.assertTrue(result["insufficient_evidence"])

    def test_unknown_background_and_dither_are_preserved(self):
        row = frame(background_delta_percent=np.nan, dither_during_exposure=pd.NA)
        event = summarize_event(pd.DataFrame([row]))
        self.assertIsNone(event["flags"]["dither_overlap"])
        self.assertTrue(event["possible_causes"]["insufficient_evidence"])
        json.dumps(event, allow_nan=False)

    def test_event_causes_require_evidence_on_the_same_frame(self):
        rows = [frame(high_guiding_rms=False), frame(high_fwhm=False)]
        event = summarize_event(pd.DataFrame(rows))
        self.assertFalse(event["possible_causes"]["possible_guiding_related"])
        self.assertTrue(event["flags"]["fwhm_degraded"])
        self.assertTrue(event["flags"]["guiding_degraded"])

    def test_missing_columns_and_empty_input(self):
        event = summarize_event(pd.DataFrame([{"filename": "unknown.fit", "is_anomaly": True}]))
        self.assertTrue(event["possible_causes"]["insufficient_evidence"])
        self.assertIsNone(event["event_start"])
        json.dumps(event, allow_nan=False)
        self.assertEqual(group_anomaly_events(pd.DataFrame()), [])
        with self.assertRaises(ValueError):
            summarize_event(pd.DataFrame())


class EventTests(unittest.TestCase):
    def rows(self):
        rows = []
        for number, minute in enumerate([0, 5, 10, 40]):
            start = pd.Timestamp("2026-01-10T00:00:00Z") + pd.Timedelta(minutes=minute)
            rows.append(frame(filename=f"light_{number}.fit", exposure_start=start,
                              exposure_end=start + pd.Timedelta(minutes=5), fwhm_delta_percent=40 + number * 10))
        return pd.DataFrame(rows)

    def test_grouping_summary_math_and_session_boundaries(self):
        events = group_anomaly_events(self.rows(), max_gap_seconds=600)
        self.assertEqual([event["frame_count"] for event in events], [3, 1])
        first = events[0]
        self.assertEqual(first["event_start"], "2026-01-10T00:00:00+00:00")
        self.assertEqual(first["event_end"], "2026-01-10T00:15:00+00:00")
        self.assertEqual(first["duration_seconds"], 900)
        self.assertEqual(first["filenames"], ["light_0.fit", "light_1.fit", "light_2.fit"])
        self.assertEqual(first["image_evidence"]["median_fwhm_delta_percent"], 50)
        self.assertEqual(first["image_evidence"]["max_fwhm_delta_percent"], 60)
        self.assertEqual(first["guiding_evidence"]["guide_sample_count_total"], 30)
        self.assertEqual(first["context"]["before"]["frame_count"], 0)
        self.assertEqual(events[-1]["context"]["after"]["frame_count"], 0)
        self.assertEqual([event["frame_count"] for event in group_anomaly_events(self.rows())], [3, 1])
        self.assertEqual(json.dumps(events, allow_nan=False), json.dumps(group_anomaly_events(self.rows()), allow_nan=False))

    def test_normal_frames_are_context_not_event_members(self):
        rows = self.rows()
        rows.loc[[1, 2], ["high_fwhm", "high_guiding_rms", "is_anomaly"]] = False
        events = group_anomaly_events(rows, max_gap_seconds=3000)
        self.assertEqual(events[0]["frame_count"], 2)
        self.assertEqual(events[0]["filenames"], ["light_0.fit", "light_3.fit"])
        separate = group_anomaly_events(rows, max_gap_seconds=600)
        self.assertEqual(len(separate), 2)
        self.assertEqual(separate[0]["context"]["after"]["frame_count"], 2)
        self.assertEqual(separate[1]["context"]["before"]["frame_count"], 2)

    def test_undated_frames_are_separate_and_bad_gaps_are_rejected(self):
        rows = self.rows().iloc[:2].copy()
        rows[["exposure_start", "exposure_end", "date_observed"]] = None
        events = group_anomaly_events(rows)
        self.assertEqual(len(events), 2)
        self.assertIsNone(events[0]["duration_seconds"])
        for gap in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                group_anomaly_events(rows, max_gap_seconds=gap)

    def test_trends(self):
        for values, expected in (([1, 2, 3], "increasing"), ([3, 2, 1], "decreasing"),
                                 ([2, 2, 2], "stable"), ([1, 2], "insufficient_data")):
            rows = pd.DataFrame({"date_observed": pd.date_range("2026-01-10", periods=len(values), freq="5min"),
                                 "median_fwhm_px": values})
            self.assertEqual(calculate_session_trends(rows)["fwhm"]["trend"], expected)
