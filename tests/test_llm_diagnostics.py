import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

from astroframe.diagnostics import group_anomaly_events
from astroframe.llm import ask_model, check_ollama_connection, diagnose_event, diagnose_events, validate_diagnosis
from astroframe.prompts import build_event_diagnosis_prompt
from config import REASONING_MODEL
from test_diagnostics import frame


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.event = group_anomaly_events(pd.DataFrame([frame()]))[0]
        self.diagnosis = {
            "classification": "possible_guiding_related", "confidence": "moderate",
            "summary": "Image degradation coincided with worse guiding.",
            "evidence": ["FWHM and guiding RMS increased while background stayed stable."],
            "alternatives": ["Wind or vibration could cause similar tracking symptoms."],
            "recommendations": ["Review guiding telemetry during this interval."],
        }

    def test_prompt_contains_only_event_evidence_and_rules(self):
        prompt = build_event_diagnosis_prompt(self.event)
        for value in ("Use ONLY", "Do not recalculate", "possible_causes", "possible_guiding_related", "3.0", "classification"):
            self.assertIn(value, prompt)
        for value in ("OLLAMA_HOST", "file_path", "raw_pixels"):
            self.assertNotIn(value, prompt)
        extra = {**self.event, "raw_pixels": [[1, 2]]}
        with self.assertRaises(ValueError):
            build_event_diagnosis_prompt(extra)

    def test_valid_json_and_grounded_numbers(self):
        self.assertEqual(validate_diagnosis(json.dumps(self.diagnosis), self.event), self.diagnosis)
        self.diagnosis["evidence"] = ["FWHM was 50% above its baseline."]
        validate_diagnosis(json.dumps(self.diagnosis), self.event)
        self.diagnosis["evidence"] = ["Wind speed reached 9876 mph."]
        with self.assertRaisesRegex(ValueError, "number absent"):
            validate_diagnosis(json.dumps(self.diagnosis), self.event)

    def test_invalid_json_fields_confidence_and_classification(self):
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            validate_diagnosis("probably guiding", self.event)
        cases = [
            {key: value for key, value in self.diagnosis.items() if key != "confidence"},
            {**self.diagnosis, "confidence": "100%"},
            {**self.diagnosis, "classification": "possible_transparency_related"},
            {**self.diagnosis, "evidence": "not a list"},
            {**self.diagnosis, "summary": ""},
            {**self.diagnosis, "thinking": "private reasoning"},
        ]
        for diagnosis in cases:
            with self.subTest(diagnosis=diagnosis), self.assertRaises(ValueError):
                validate_diagnosis(json.dumps(diagnosis), self.event)

    @patch("astroframe.llm.requests.post")
    def test_chat_uses_json_schema_and_ignores_thinking(self, post):
        post.return_value = Mock(json=Mock(return_value={"message": {
            "content": json.dumps(self.diagnosis), "thinking": "not stored",
        }}))
        result = diagnose_event(self.event)
        self.assertEqual(result["event"], self.event)
        self.assertEqual(result["diagnosis"], self.diagnosis)
        self.assertNotIn("not stored", json.dumps(result))
        payload = post.call_args.kwargs["json"]
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"]["type"], "object")
        self.assertEqual(payload["options"]["temperature"], 0)

    @patch("astroframe.llm.requests.post")
    def test_chat_timeout_connection_and_bad_content(self, post):
        for error, message in ((requests.Timeout(), "timed out"), (requests.ConnectionError(), "Unable to reach")):
            post.side_effect = error
            with self.assertRaisesRegex(RuntimeError, message):
                ask_model("test")
        post.side_effect = None
        post.return_value = Mock(json=Mock(return_value={"message": {"content": "bad JSON"}}))
        with self.assertRaises(ValueError):
            diagnose_event(self.event)
        post.return_value = Mock(json=Mock(return_value={"bad": "response"}))
        with self.assertRaisesRegex(RuntimeError, "invalid chat"):
            ask_model("test")

    @patch("astroframe.llm.requests.get")
    def test_health_check_model_missing_and_unreachable(self, get):
        get.return_value = Mock(json=Mock(return_value={"models": [{"name": REASONING_MODEL}]}))
        self.assertTrue(check_ollama_connection()["model_available"])
        get.return_value = Mock(json=Mock(return_value={"models": []}))
        result = check_ollama_connection()
        self.assertTrue(result["reachable"])
        self.assertFalse(result["model_available"])
        self.assertIn(f"ollama pull {REASONING_MODEL}", result["error"])
        for error in (requests.ConnectionError(), requests.Timeout()):
            get.side_effect = error
            result = check_ollama_connection()
            self.assertFalse(result["reachable"])
            self.assertIn("error", result)

    @patch("astroframe.llm.check_ollama_connection", return_value={"reachable": True, "model_available": True})
    @patch("astroframe.llm.diagnose_event")
    def test_failed_event_keeps_completed_results(self, diagnose, health):
        events = [{**self.event, "event_id": index} for index in (1, 2, 3)]
        diagnose.side_effect = [
            {"event": events[0], "diagnosis": self.diagnosis}, RuntimeError("Ollama request timed out."),
            {"event": events[2], "diagnosis": self.diagnosis},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnoses.json"
            with contextlib.redirect_stdout(io.StringIO()), self.assertLogs("astroframe.llm", level="WARNING"):
                results = diagnose_events(events, output_path=path)
            saved = json.loads(path.read_text())
        self.assertEqual(len(saved["events"]), 3)
        self.assertEqual(saved["events"], results)
        self.assertIn("error", results[1])
        self.assertIsNotNone(results[2]["diagnosis"])
