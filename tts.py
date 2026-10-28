import subprocess
import sys
import os


_speech = None
_last_text = ""
VOICE=os.environ.get("TTS_VOICE", "mb-us3")
SPEED = os.environ.get("TTS_SPEED", "150")
AMPLITUDE = os.environ.get("TTS_AMPLITUDE", "100")

def _tts_command(text):
    if sys.platform.startswith("linux"):
        return ["espeak-ng", "-v", VOICE, "-s", SPEED, "-a", AMPLITUDE, "--", text]
    # Windows/macOS: pyttsx3 in a child process, so "stop" can kill it
    return [sys.executable, "-c",
            "import sys, pyttsx3; e = pyttsx3.init(); e.say(sys.argv[1]); e.runAndWait()",
            text]


def speak(text, force=False):
    global _speech, _last_text
    text = text.strip()
    if not text or (text == _last_text and not force):
        return
    if _speech is not None and _speech.poll() is None:
        if not force:
            return
        _speech.terminate()
    _last_text = text
    try:
        _speech = subprocess.Popen(_tts_command(text))
    except FileNotFoundError:
        print("[TTS] espeak-ng is not installed (sudo apt install espeak-ng)")


def stop_speaking():
    global _speech
    if _speech is not None and _speech.poll() is None:
        _speech.terminate()
    _speech = None
