import json
import logging
from pathlib import Path
import re

import requests

from config import OLLAMA_HOST, REASONING_MODEL
from astroframe.prompts import allowed_classifications, build_event_diagnosis_prompt, diagnosis_schema

logger = logging.getLogger(__name__)


def ask_model(prompt: str, *, response_format: dict | None = None) -> str:
    """Send a prompt to the configured Ollama model."""

    payload = {
        "model": REASONING_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if response_format is not None:
        payload.update({"format": response_format, "options": {"temperature": 0}})
    try:
        response = requests.post(f"{OLLAMA_HOST.rstrip('/')}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        content = response.json()["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Missing final content")
        return content
    except requests.Timeout as exc:
        raise RuntimeError("Ollama request timed out.") from exc
    except requests.RequestException as exc:
        raise RuntimeError("Unable to reach Ollama at the configured host.") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Ollama returned an invalid chat response.") from exc


def check_ollama_connection() -> dict:
    """Check whether Ollama and the configured model are available."""
    status = {"reachable": False, "model_available": False, "model": REASONING_MODEL}
    try:
        response = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        status["reachable"] = True
        models = response.json()["models"]
        if not isinstance(models, list):
            raise ValueError("Invalid model list")
        names = {model.get("name") for model in models if isinstance(model, dict)}
        name = REASONING_MODEL if ":" in REASONING_MODEL else f"{REASONING_MODEL}:latest"
        status["model_available"] = REASONING_MODEL in names or name in names
        if not status["model_available"]:
            status["error"] = "Configured Ollama model is not installed."
    except requests.Timeout:
        status["error"] = "Ollama connection check timed out."
    except requests.RequestException:
        status["error"] = "Unable to reach Ollama at the configured host."
    except (ValueError, KeyError, TypeError):
        status["error"] = "Ollama returned an invalid model list."
    return status


def validate_diagnosis(content: str, event: dict) -> dict:
    """Reject malformed output, unsupported causes, and unfamiliar numbers."""
    try:
        diagnosis = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise ValueError("Diagnosis was not valid JSON.") from exc
    required = set(diagnosis_schema(event)["required"])
    if not isinstance(diagnosis, dict) or set(diagnosis) != required:
        raise ValueError("Diagnosis has missing or unexpected fields.")
    if diagnosis["classification"] not in allowed_classifications(event):
        raise ValueError("Diagnosis classification is not supported by the event evidence.")
    if diagnosis["confidence"] not in ("low", "moderate", "high"):
        raise ValueError("Diagnosis confidence must be low, moderate, or high.")
    if not isinstance(diagnosis["summary"], str) or not diagnosis["summary"].strip():
        raise ValueError("Diagnosis summary must be a nonempty string.")
    text = [diagnosis["summary"]]
    for name in ("evidence", "alternatives", "recommendations"):
        if not isinstance(diagnosis[name], list) or not all(isinstance(item, str) for item in diagnosis[name]):
            raise ValueError(f"Diagnosis {name} must be a list of strings.")
        text.extend(diagnosis[name])
    numbers = r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])"
    supplied = [float(value) for value in re.findall(numbers, json.dumps(event, allow_nan=False))]
    for claim in re.findall(numbers, " ".join(text)):
        places = len(claim.partition(".")[2]) if "." in claim else 0
        if not any(float(claim) == round(value, places) or float(claim) == value for value in supplied):
            raise ValueError("Diagnosis introduced a number absent from the event evidence.")
    return diagnosis


def diagnose_event(event: dict) -> dict:
    """Explain one event and keep its measurements alongside the result."""
    prompt = build_event_diagnosis_prompt(event)
    content = ask_model(prompt, response_format=diagnosis_schema(event))
    return {"event": event, "diagnosis": validate_diagnosis(content, event)}


def _save_diagnoses(results: list[dict], output_path: str | Path | None) -> None:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": REASONING_MODEL, "events": results}, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def diagnose_events(events: list[dict], *, output_path: str | Path | None = None) -> list[dict]:
    """Diagnose events in order and save progress after each result."""
    results = []
    status = check_ollama_connection() if events else {}
    _save_diagnoses(results, output_path)
    for number, event in enumerate(events, 1):
        print(f"[{number}/{len(events)}] Diagnosing event {event['event_id']}")
        try:
            if status.get("error"):
                raise RuntimeError(status["error"])
            result = diagnose_event(event)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Event %s diagnosis failed: %s", event["event_id"], exc)
            result = {"event": event, "diagnosis": None, "error": str(exc)}
        results.append(result)
        _save_diagnoses(results, output_path)
    return results
