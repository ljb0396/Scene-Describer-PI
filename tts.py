"""Text-to-speech helper: speak(text, force=False), stop_speaking(), warm_up().

Linux (the Pi), in order of preference:
  1. Piper neural voice (default en_US-lessac-medium): the `piper` program
     synthesizes raw audio and `aplay` plays it. Both run as child processes,
     so stop_speaking() can kill them instantly.
  2. espeak-ng, if Piper or its voice files are missing.
Windows/macOS: pyttsx3 in a child process.

Expected files (paths relative to this folder, overridable):
    piper/piper                           the Piper program (arm64 release)
    voices/en_US-lessac-medium.onnx       the voice
    voices/en_US-lessac-medium.onnx.json  its config (must sit next to the .onnx)

Environment variables (all optional):
    TTS_ENGINE        auto (default) | piper | espeak
    PIPER_BIN         path to the piper program
    PIPER_MODEL       path to the voice .onnx file
    TTS_ALSA_DEVICE   aplay output device, e.g. plughw:1,0 (default: system default)
    TTS_VOICE         espeak-ng voice for the fallback (default: en-us)
    TTS_SPEED         espeak-ng words per minute (default: 160)
    TTS_AMPLITUDE     espeak-ng volume, 0-200 (default: 100)
"""

import json
import os
import shlex
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

IS_LINUX = sys.platform.startswith("linux")

PIPER_BIN = os.environ.get("PIPER_BIN", os.path.join(_HERE, "piper", "piper"))
PIPER_MODEL = os.environ.get(
    "PIPER_MODEL", os.path.join(_HERE, "voices", "en_US-lessac-low.onnx"))
ALSA_DEVICE = os.environ.get("TTS_ALSA_DEVICE", "")
# Optional: replace aplay with any player that reads raw 16-bit mono audio on stdin.
# {rate} is filled in from the voice's config. Example for PipeWire/Bluetooth:
#   TTS_PLAYER_CMD="paplay --raw --rate={rate} --format=s16le --channels=1"
PLAYER_CMD = os.environ.get("TTS_PLAYER_CMD", "")

ESPEAK_VOICE_WANTED = os.environ.get("TTS_VOICE", "en-us")
SPEED = os.environ.get("TTS_SPEED", "160")
AMPLITUDE = os.environ.get("TTS_AMPLITUDE", "100")

_procs = []          # child processes of the utterance being spoken (last one = the player)
_last_text = ""


def _missing_for_piper():
    """List what is missing for Piper; an empty list means it can be used."""
    missing = []
    if not (os.path.isfile(PIPER_BIN) and os.access(PIPER_BIN, os.X_OK)):
        missing.append(f"program {PIPER_BIN}")
    if not os.path.isfile(PIPER_MODEL):
        missing.append(f"voice {PIPER_MODEL}")
    elif not os.path.isfile(PIPER_MODEL + ".json"):
        missing.append(f"config {PIPER_MODEL}.json")
    if PLAYER_CMD:
        player = shlex.split(PLAYER_CMD)[0]
        if shutil.which(player) is None:
            missing.append(f"player {player} (from TTS_PLAYER_CMD)")
    elif shutil.which("aplay") is None:
        missing.append("aplay (sudo apt install alsa-utils)")
    return missing


def _pick_espeak_voice():
    wanted = ESPEAK_VOICE_WANTED
    if wanted.startswith("mb-"):                             # MBROLA voices need extra packages
        name = wanted[3:]
        if shutil.which("mbrola") is None or not os.path.exists(f"/usr/share/mbrola/{name}/{name}"):
            print(f"[TTS] espeak voice '{wanted}' is not installed; using en-us")
            return "en-us"
    return wanted


def _sample_rate():
    try:
        with open(PIPER_MODEL + ".json") as f:
            return int(json.load(f)["audio"]["sample_rate"])
    except Exception:
        return 22050                                         # lessac-medium


def _choose_engine():
    if not IS_LINUX:
        return "pyttsx3"
    want = os.environ.get("TTS_ENGINE", "auto").lower()
    if want == "espeak":
        return "espeak"
    missing = _missing_for_piper()
    if not missing:
        return "piper"
    print(f"[TTS] Piper not available (missing: {'; '.join(missing)}); using espeak-ng")
    return "espeak"


ENGINE = _choose_engine()
ESPEAK_VOICE = _pick_espeak_voice() if ENGINE == "espeak" else None
if ENGINE == "piper":
    print(f"[TTS] using Piper voice {os.path.basename(PIPER_MODEL)}")


def _aplay_cmd():
    if PLAYER_CMD:
        return shlex.split(PLAYER_CMD.format(rate=_sample_rate()))
    cmd = ["aplay", "-q", "-t", "raw", "-r", str(_sample_rate()), "-f", "S16_LE", "-c", "1"]
    if ALSA_DEVICE:
        cmd += ["-D", ALSA_DEVICE]
    return cmd + ["-"]


def _start_piper(text):
    """Start piper -> aplay for one utterance. Returns [piper, aplay]."""
    piper = subprocess.Popen([PIPER_BIN, "--model", PIPER_MODEL, "--output-raw"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
    try:
        player = subprocess.Popen(_aplay_cmd(), stdin=piper.stdout,
                                  stderr=subprocess.DEVNULL)
    except Exception:
        piper.kill()
        raise
    piper.stdout.close()                                     # piper gets SIGPIPE if aplay dies
    try:
        piper.stdin.write((" ".join(text.split()) + "\n").encode("utf-8"))
        piper.stdin.close()
    except BrokenPipeError:
        pass
    return [piper, player]


def _start(text):
    if ENGINE == "piper":
        return _start_piper(text)
    if ENGINE == "espeak":
        return [subprocess.Popen(["espeak-ng", "-v", ESPEAK_VOICE, "-s", SPEED,
                                  "-a", AMPLITUDE, "--", text])]
    # Windows/macOS: pyttsx3 in a child process, so "stop" can kill it
    return [subprocess.Popen([
        sys.executable, "-c",
        "import sys, pyttsx3; e = pyttsx3.init(); e.say(sys.argv[1]); e.runAndWait()",
        text])]


def _kill(procs):
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=1)
        except subprocess.TimeoutExpired:
            p.kill()


def _busy():
    return bool(_procs) and _procs[-1].poll() is None


def speak(text, force=False):
    """Say `text`. Ignores empty text, repeats, and anything said while still
    talking, unless force=True (which interrupts the current speech)."""
    global _procs, _last_text
    text = text.strip()
    if not text or (text == _last_text and not force):
        return
    if _busy():
        if not force:
            return
        _kill(_procs)
    _last_text = text
    try:
        _procs = _start(text)
    except FileNotFoundError as e:
        print(f"[TTS] could not start speech: {e}")
        _procs = []


def stop_speaking():
    global _procs
    _kill(_procs)
    _procs = []


def warm_up():
    """Optional: call once at startup (in a background thread). Loads Piper and its
    voice once so the operating system caches the files, making the first real
    answer start sooner. Produces no sound."""
    if ENGINE != "piper":
        return
    try:
        p = subprocess.Popen([PIPER_BIN, "--model", PIPER_MODEL, "--output-raw"],
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        p.communicate(b"Ready.\n", timeout=60)
    except Exception as e:
        print(f"[TTS] warm-up failed: {e}")
