"""Single audio-feedback channel shared by every task (YOLO, OCR, describe).

Why this exists:
  * pyttsx3's runAndWait() blocks and cannot be interrupted, so "stop" had to
    wait for the current sentence. Here speech is a subprocess we can kill.
  * Every task used to build its own TTS engine. Now there is one worker
    thread, one queue, and one place to tune voice/volume/rate.
  * Short tones ("earcons") tell the user what the system is doing without
    spending seconds on speech.

Backends (first one found wins, override with TTS_BACKEND):
  espeak-ng / espeak  -> Raspberry Pi (apt install espeak-ng), ~2 MB RAM
  say                 -> macOS dev machine
  pyttsx3             -> last resort, NOT interruptible mid-sentence

Env vars: TTS_BACKEND, TTS_RATE (words/min, default 160), TTS_VOICE,
          AUDIO_CUES=0 to disable tones.
"""

import heapq
import itertools
import os
import shutil
import subprocess
import threading
import time

import numpy as np

# Lower number = spoken first.
PRIORITY_ALERT = 0    # command acknowledgements, errors
PRIORITY_NORMAL = 1   # scene descriptions, OCR text
PRIORITY_LOW = 2      # chatty per-object YOLO announcements

MAX_QUEUE = 6                 # drop LOW items rather than build a backlog
REPEAT_COOLDOWN = 4.0         # seconds before the same text is spoken again
CUE_SAMPLE_RATE = 22050

# name -> list of (frequency_hz, seconds). Distinct shapes so they are
# recognisable by ear: rising = start, falling = stop, low buzz = error.
CUES = {
    "listening": [(880, 0.08)],
    "start": [(660, 0.07), (880, 0.09)],
    "thinking": [(520, 0.06), (520, 0.06)],
    "done": [(880, 0.07), (1175, 0.10)],
    "stop": [(880, 0.07), (520, 0.10)],
    "error": [(180, 0.25)],
}


def _tone(freq, seconds, volume=0.35):
    t = np.linspace(0, seconds, int(CUE_SAMPLE_RATE * seconds), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    fade = min(len(wave) // 4, 200)          # avoid clicks
    if fade:
        ramp = np.linspace(0, 1, fade)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return (wave * volume * 32767).astype(np.int16)


class _CommandBackend:
    """Speak by piping text to an external TTS process (killable)."""

    def __init__(self, argv):
        self.argv = argv
        self._proc = None
        self._lock = threading.Lock()

    def speak(self, text):
        with self._lock:
            self._proc = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc = self._proc
        try:
            # Text goes over stdin, never argv, so OCR output that starts
            # with "-" or contains shell characters cannot be misparsed.
            proc.communicate(text.encode("utf-8", "replace"))
        except (BrokenPipeError, OSError):
            pass

    def stop(self):
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()


class _Pyttsx3Backend:
    def __init__(self, rate):
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)

    def speak(self, text):
        self._engine.say(text)
        self._engine.runAndWait()

    def stop(self):
        try:
            self._engine.stop()
        except Exception:
            pass


def _pick_backend():
    rate = int(os.environ.get("TTS_RATE", "160"))
    voice = os.environ.get("TTS_VOICE", "mb-us1")
    want = os.environ.get("TTS_BACKEND", "espeak-ng").lower()

    for name in ("espeak-ng", "espeak"):
        if want in ("", name) and shutil.which(name):
            argv = [name, "--stdin", "-s", str(rate)]
            if voice:
                argv += ["-v", voice]
            return _CommandBackend(argv)

    if want in ("", "say") and shutil.which("say"):
        argv = ["say", "-r", str(rate), "-f", "-"]
        if voice:
            argv += ["-v", voice]
        return _CommandBackend(argv)

    return _Pyttsx3Backend(rate)


class Speaker:
    def __init__(self, backend=None):
        self._backend = backend or _pick_backend()
        self._cues_enabled = os.environ.get("AUDIO_CUES", "1") != "0"
        self._heap = []
        self._seq = itertools.count()
        self._cond = threading.Condition()
        self._last_spoken = {}          # text -> monotonic time
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---- public API -------------------------------------------------

    def speak(self, text, priority=PRIORITY_NORMAL, interrupt=False):
        """Queue text. interrupt=True cuts off whatever is being said."""
        text = " ".join((text or "").split())
        if not text:
            return
        if interrupt:
            self.stop()
        with self._cond:
            now = time.monotonic()
            if now - self._last_spoken.get(text, -1e9) < REPEAT_COOLDOWN:
                return
            if any(item[3] == text for item in self._heap):
                return
            if len(self._heap) >= MAX_QUEUE:
                if priority >= PRIORITY_LOW:
                    return
                self._drop_lowest()
            heapq.heappush(self._heap, (priority, next(self._seq), "speech", text))
            self._cond.notify()

    def cue(self, name, interrupt=False):
        if not self._cues_enabled or name not in CUES:
            return
        if interrupt:
            self.stop()
        with self._cond:
            heapq.heappush(
                self._heap, (PRIORITY_ALERT, next(self._seq), "cue", name)
            )
            self._cond.notify()

    def stop(self):
        """Clear everything queued and cut off the current utterance."""
        with self._cond:
            self._heap.clear()
        self._backend.stop()

    def wait_idle(self, timeout=None):
        """Block until the queue is drained (used by tests/scripts)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._cond:
                if not self._heap and not self._busy:
                    return True
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.05)

    def close(self):
        self._closed = True
        self.stop()
        with self._cond:
            self._cond.notify_all()

    # ---- internals --------------------------------------------------

    _busy = False

    def _drop_lowest(self):
        worst = max(self._heap)
        self._heap.remove(worst)
        heapq.heapify(self._heap)

    def _run(self):
        while not self._closed:
            with self._cond:
                while not self._heap and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                _, _, kind, payload = heapq.heappop(self._heap)
                self._busy = True
            try:
                if kind == "speech":
                    print(f"[Audio] Speaking: {payload}")
                    self._backend.speak(payload)
                    with self._cond:
                        self._last_spoken[payload] = time.monotonic()
                else:
                    self._play_cue(payload)
            except Exception as e:           # never let audio kill a task
                print(f"[Audio Error]: {e}")
            finally:
                self._busy = False

    def _play_cue(self, name):
        try:
            import sounddevice as sd
        except Exception:
            return
        samples = np.concatenate([_tone(f, s) for f, s in CUES[name]])
        sd.play(samples, CUE_SAMPLE_RATE)
        sd.wait()


_speaker = None
_speaker_lock = threading.Lock()


def get_speaker():
    """Process-wide Speaker, created on first use."""
    global _speaker
    with _speaker_lock:
        if _speaker is None:
            _speaker = Speaker()
        return _speaker


def speak(text, priority=PRIORITY_NORMAL, interrupt=False):
    get_speaker().speak(text, priority, interrupt)


def cue(name, interrupt=False):
    get_speaker().cue(name, interrupt)


def stop():
    get_speaker().stop()


if __name__ == "__main__":
    # Quick hardware check: python audio_feedback.py
    s = get_speaker()
    for name in CUES:
        print("cue:", name)
        s.cue(name)
    s.wait_idle(10)                      # let the tones finish
    s.speak("Audio check. This is a long sentence that should be cut off.")
    time.sleep(1.5)
    s.stop()
    s.speak("Interrupted cleanly.", PRIORITY_ALERT)
    s.wait_idle(15)
