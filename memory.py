"""Short-term conversation memory for the cloud vision model.

The cloud API is stateless, so "memory" means resending recent exchanges with
each request. We keep TEXT ONLY (what the user asked and what we answered),
never old images, so history costs a few hundred bytes per turn instead of a
full JPEG.

Memory lives in RAM only: it is lost on restart, expires after a period of
inactivity, and can be cleared on demand (the "forget" voice command).

Settings (environment variables, all optional):
    MEMORY_TURNS   how many past exchanges to keep   (default 6, 0 = off)
    MEMORY_TTL_S   seconds of inactivity before it is forgotten (default 300)
"""

import os
import threading
import time

MAX_TURNS = int(os.environ.get("MEMORY_TURNS", "6"))
TTL_S = float(os.environ.get("MEMORY_TTL_S", "300"))
MAX_CHARS = 300                      # long answers (e.g. a read sign) are truncated

_lock = threading.Lock()
_turns = []                          # list of (user_text, answer)
_last_used = 0.0


def _expire():
    """Caller must hold the lock."""
    global _last_used
    if _turns and time.monotonic() - _last_used > TTL_S:
        _turns.clear()


def history():
    """Past exchanges as chat messages ([] if memory is empty, off, or expired)."""
    if MAX_TURNS <= 0:
        return []
    with _lock:
        _expire()
        messages = []
        for user_text, answer in _turns:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": answer})
        return messages


def add(user_text, answer):
    """Remember one exchange (call after a successful answer)."""
    global _last_used
    if MAX_TURNS <= 0 or not answer:
        return
    with _lock:
        _expire()
        _turns.append((user_text[:MAX_CHARS], answer[:MAX_CHARS]))
        del _turns[:-MAX_TURNS]      # keep only the newest MAX_TURNS
        _last_used = time.monotonic()


def clear():
    with _lock:
        _turns.clear()
