"""Thin Ollama client for scene description. Uses only requests + cv2, which
are already dependencies, so nothing extra is loaded on the Pi.

Two ways to reach an Ollama cloud model:

  1. Direct to the cloud API (no daemon on the Pi, lowest RAM):
        OLLAMA_HOST=https://ollama.com        (default)
        OLLAMA_API_KEY=<key from ollama.com/settings/keys>
        OLLAMA_MODEL=<a vision-capable cloud model, no "-cloud" suffix>
        OLLAMA_THINK=1 to allow thinking (default off; can empty the reply)

  2. Through a local Ollama daemon that has run `ollama signin`
     (also works for a LAN machine running Ollama):
        OLLAMA_HOST=http://localhost:11434
        OLLAMA_MODEL=<model>-cloud
     No API key needed in this mode.

Model names change; list current ones with `curl https://ollama.com/api/tags`
and pick one that supports vision.

describe_frame() accepts optional `system` and `max_tokens` arguments so
callers (for example the text-reading path) can override the default
two-sentence scene-description behavior.
"""

import base64
import os
import time

import cv2
import requests

DEFAULT_HOST = "https://ollama.com"
DEFAULT_MODEL = "gemma4:31b"

# Keep uploads small: slow Wi-Fi + Pi CPU encode cost. 640px is plenty for a
# scene summary and stays ~30-60 KB as JPEG.
MAX_SIDE = int(os.environ.get("OLLAMA_MAX_SIDE", "640"))
JPEG_QUALITY = int(os.environ.get("OLLAMA_JPEG_QUALITY", "70"))

SYSTEM_PROMPT = (
    "You describe camera images for a blind or low-vision person who hears "
    "your answer through a speaker. Reply in at most two short plain "
    "sentences. If the person asks a question, answer that question first and "
    "directly. If the answer is not visible, say so; do not guess. Mention "
    "hazards and the nearest or largest objects first, and give direction as "
    "left, right, or ahead. No markdown, lists, or preamble."
)
USER_PROMPT = "Describe what is in front of me."
DEFAULT_MAX_TOKENS = 120


class OllamaError(Exception):
    """Raised with a short, speakable message."""


def encode_frame(frame):
    """Downscale + JPEG-encode a BGR frame to base64 for the API."""
    h, w = frame.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OllamaError("Could not encode the camera image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _post(url, payload, headers, timeout):
    """POST and translate network failures into speakable OllamaErrors."""
    try:
        return requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.ConnectTimeout:
        raise OllamaError("Could not reach the cloud model in time.")
    except requests.exceptions.ReadTimeout:
        raise OllamaError("The cloud model took too long to answer.")
    except requests.exceptions.RequestException:
        raise OllamaError("No network connection to the cloud model.")


def describe_frame(frame, prompt=USER_PROMPT, timeout=(5, 45),
                   system=SYSTEM_PROMPT, max_tokens=DEFAULT_MAX_TOKENS):
    """Send one frame to the configured Ollama model; return the text reply.

    prompt     -- the user message sent with the image
    timeout    -- (connect, read) seconds for the HTTP request
    system     -- system prompt (default: short scene description)
    max_tokens -- cap on the reply length (raise it for reading long text)

    Raises OllamaError with a message safe to read aloud.
    """
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    api_key = os.environ.get("OLLAMA_API_KEY")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif host == DEFAULT_HOST:
        raise OllamaError("Ollama API key is not set.")

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt, "images": [encode_frame(frame)]},
        ],
        # Short answers = faster + cheaper + less to listen to.
        "options": {"num_predict": max_tokens, "temperature": 0.2},
    }
    # Thinking models can burn the whole token budget reasoning and return an
    # empty answer; we want a fast spoken reply, so turn thinking off.
    if os.environ.get("OLLAMA_THINK", "0") == "0":
        payload["think"] = False

    url = f"{host}/api/chat"
    started = time.monotonic()
    resp = _post(url, payload, headers, timeout)

    if resp.status_code == 400 and "think" in payload:
        # Some models reject the think flag outright; retry without it.
        del payload["think"]
        resp = _post(url, payload, headers, timeout)

    if "usage" in resp.text[:300].lower() and not resp.ok:
        raise OllamaError("This model needs paid Ollama usage. Pick another.")
    if resp.status_code in (401, 403):
        raise OllamaError("Ollama rejected the API key.")
    if resp.status_code == 404:
        raise OllamaError(f"Model {model} was not found.")
    if resp.status_code == 410:
        raise OllamaError(f"Model {model} is no longer available.")
    if resp.status_code == 429:
        raise OllamaError("Cloud usage limit reached.")
    if not resp.ok:
        raise OllamaError(f"The cloud model returned error {resp.status_code}.")

    try:
        text = resp.json()["message"]["content"].strip()
    except (ValueError, KeyError, AttributeError):
        raise OllamaError("The cloud model sent an unreadable reply.")
    if not text:
        raise OllamaError("The cloud model gave an empty answer.")

    print(f"[Ollama] {model} replied in {time.monotonic() - started:.1f}s -> {text}")
    return text
