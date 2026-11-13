import logging
import os
import time

import local_vision
import ollama_client
import memory

log = logging.getLogger("vision")

MODE = os.environ.get("VISION_MODE", "auto")             # auto | cloud | local
COOLDOWN_S = float(os.environ.get("CLOUD_COOLDOWN_S", "30"))
CLOUD_TIMEOUT = (3, 15)                                  # fail fast so the fallback feels quick

READ_SYSTEM = ("You read text from camera images for a blind person who hears the answer "
               "through a speaker. Output only the visible text in natural reading order, "
               "with no commentary. If there is no readable text, say exactly: "
               "I don't see any text.")

_down_until = 0.0
_last_source = "cloud"


def _run(kind, cloud_fn, local_fn, remember=None):
    """Returns (text, source). Announces offline/online changes by prefixing the answer."""
    global _down_until, _last_source
    text, source = None, "local"
    if MODE != "local" and time.monotonic() >= _down_until:
        try:
            text, source = cloud_fn(), "cloud"
        except Exception as e:
            if isinstance(e, ollama_client.OllamaError):
                log.warning("cloud %s failed: %s", kind, e)
            else:
                log.exception("cloud %s crashed unexpectedly", kind)   # full traceback in the log
            _down_until = time.monotonic() + COOLDOWN_S
            if MODE == "cloud":
                msg = str(e) if isinstance(e, ollama_client.OllamaError) else "The cloud model failed."
                return msg, "error"
    
    if text is None:
        text = local_fn()
    if remember:
        memory.add(remember, text)
    prefix = ""
    if MODE == "auto" and source != _last_source:
        prefix = "Back online. " if source == "cloud" else "Offline mode. "
    _last_source = source
    return prefix + text, source


def describe(frame):
    return _run("see",
                lambda: ollama_client.describe_frame(
                    frame, timeout=CLOUD_TIMEOUT, history=memory.history()),
                lambda: local_vision.describe(frame),
                remember=ollama_client.USER_PROMPT)


def read(frame):
    # No history sent: reading is self-contained, but the result is remembered
    # so a follow-up question like "what did that say about parking?" works.
    return _run("read",
                lambda: ollama_client.describe_frame(
                    frame, prompt="Read the text in this image.", system=READ_SYSTEM,
                    max_tokens=300, timeout=CLOUD_TIMEOUT),
                lambda: local_vision.read_text(frame),
                remember="Read the text in this image.")


def ask(frame, question):
    prompt = (f"The person asked: {question}\n"
              "Answer from the image. Use the earlier conversation only to "
              "understand what the person is referring to.")
    return _run("ask",
                lambda: ollama_client.describe_frame(
                    frame, prompt=prompt, timeout=CLOUD_TIMEOUT, history=memory.history()),
                lambda: local_vision.answer_question(question, frame),
                remember=question)
