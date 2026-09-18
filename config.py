import os


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

REASONING_MODEL = os.getenv(
    "REASONING_MODEL",
    "gpt-oss:20b",
)
