# Scene-Describer-PI

Voice commands (heard over UDP from the ESP32 mic): **read** (OCR), **see**
(YOLO object announcements), **describe** (Ollama cloud vision model),
**stop**.

## Run
    python test_vosk.py

## Ollama cloud ("describe")
    export OLLAMA_API_KEY=...            # ollama.com/settings/keys
    export OLLAMA_MODEL=qwen3-vl:235b    # any vision-capable cloud model
    python test_describe.py              # one-shot test, no mic needed

Or route through a local/LAN Ollama daemon: `OLLAMA_HOST=http://host:11434`,
model name with `-cloud` suffix, no key. See `ollama_client.py`.

## Audio feedback
All speech goes through `audio_feedback.py` (one queue, interruptible,
priority-ordered, plus short tones). Check the speaker with
`python audio_feedback.py`. Env: `TTS_BACKEND`, `TTS_RATE`, `TTS_VOICE`,
`AUDIO_CUES=0`. Other: `YOLO_MAX_FPS` (default 3), `VIDEO_SOURCE`.

Pi install: see `requirements-pi.txt`.

## Camera
The ESP32-CAM is upside down in some mounts: set `VIDEO_ROTATE=180` (0/90/180/270).
`view_camera.py` shows a live preview window; from a dev machine tunnel through
the Pi first (`ssh -N -L 8081:192.168.4.1:80 user@pi`, then
`VIDEO_SOURCE=http://localhost:8081/stream`).

## Secrets
Keep `OLLAMA_API_KEY` out of the repo: put it in a `chmod 600` env file
(e.g. `~/.config/scene-describer.env`) and `set -a; . that-file; set +a`.
