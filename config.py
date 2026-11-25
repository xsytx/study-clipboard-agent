"""Load local credentials, enable LangSmith, and construct the LLM."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "openai/gpt-oss-120b"


def load_environment() -> None:
    load_dotenv(PROJECT_DIR / ".env", override=False)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "Study & Clipboard Agent")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")


def missing_credentials() -> list[str]:
    load_environment()
    placeholders = {"", "your_groq_api_key_here", "your_langsmith_api_key_here"}
    return [
        name for name in ("GROQ_API_KEY", "LANGSMITH_API_KEY")
        if os.getenv(name, "").strip() in placeholders
    ]


@lru_cache(maxsize=4)
def _create_llm(model: str, api_key: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )


def get_llm(model: str = DEFAULT_MODEL) -> ChatGroq:
    load_environment()
    missing = missing_credentials()
    if missing:
        raise ValueError("Add these credentials to .env: " + ", ".join(missing))
    selected_model = os.getenv("LLM_MODEL", "").strip() or model
    return _create_llm(selected_model, os.environ["GROQ_API_KEY"].strip())