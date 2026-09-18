import requests

from config import OLLAMA_HOST, REASONING_MODEL


def ask_model(prompt: str) -> str:
    """Send a prompt to the configured Ollama model."""

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": REASONING_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]
