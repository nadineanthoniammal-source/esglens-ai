"""LLM access (Groq via LangChain).

Groq changes its model catalog often, so instead of hard-coding one model this file asks
your Groq account which models it can use and picks the best available one.
"""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

# Best first. The first one your account has is used.
PREFERRED = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "llama-3.1-8b-instant",
]
# Not chat models (speech, safety filters, etc.)
SKIP = ("whisper", "tts", "guard", "orpheus", "compound", "embed", "safeguard")

_models_cache: dict = {}


def _key(api_key: str | None = None) -> str:
    return api_key or os.getenv("GROQ_API_KEY", "")


def available_models(api_key: str | None = None) -> list[str]:
    from groq import Groq

    key = _key(api_key)
    if key not in _models_cache:
        _models_cache[key] = [m.id for m in Groq(api_key=key).models.list().data]
    return _models_cache[key]


def pick_model(api_key: str | None = None) -> str:
    ids = available_models(api_key)
    wanted = os.getenv("GROQ_MODEL", "").strip()
    if wanted and wanted in ids:
        return wanted
    for m in PREFERRED:
        if m in ids:
            return m
    usable = [i for i in ids if not any(s in i.lower() for s in SKIP)]
    if usable:
        return usable[0]
    raise RuntimeError(f"No usable chat model found on this Groq account. Models seen: {ids}")


def get_llm(json_mode: bool = False, api_key: str | None = None):
    from langchain_groq import ChatGroq

    kwargs = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatGroq(model=pick_model(api_key), temperature=0, api_key=_key(api_key), **kwargs)


def call_llm(prompt: str, json_mode: bool = False, api_key: str | None = None) -> str:
    text = get_llm(json_mode, api_key).invoke(prompt).content
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # drop reasoning tags if any


def parse_json(text: str) -> dict:
    """Parse JSON from an LLM reply, tolerating code fences / extra text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise
