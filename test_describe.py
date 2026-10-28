import time

import audio_feedback as audio
from ollama_client import OllamaError, describe_frame
from video_source import open_video_source

FRAME_WAIT = 5.0   # seconds to wait for the camera to deliver a frame


def describe_process(cancel_event):
    """One-shot: grab a fresh frame, ask the cloud model, speak the answer.

    Runs on the same dispatch_new_task() thread as OCR/YOLO, so saying
    "stop" cancels it. The camera is released before the network call so the
    ESP32 (which serves one client at a time) is free for the next command.
    """

    audio.cue("start")
    cap = None
    frame = None

    try:
        cap = open_video_source()
        if not cap or not cap.isOpened():
            raise OllamaError("Camera is not available.")

        deadline = time.monotonic() + FRAME_WAIT
        while frame is None and time.monotonic() < deadline:
            if cancel_event.is_set():
                return
            ok, candidate = cap.read()
            if ok:
                frame = candidate
            else:
                time.sleep(0.1)
        if frame is None:
            raise OllamaError("Camera is not sending images.")
    except OllamaError as e:
        print(f"[Describe Error]: {e}")
        audio.cue("error")
        audio.speak(str(e), audio.PRIORITY_ALERT)
        return
    finally:
        if cap is not None:
            cap.release()

    if cancel_event.is_set():
        return

    audio.cue("thinking")
    try:
        text = describe_frame(frame)
    except OllamaError as e:
        print(f"[Describe Error]: {e}")
        audio.cue("error")
        audio.speak(str(e), audio.PRIORITY_ALERT)
        return
    except Exception as e:
        print(f"[Describe Error]: {e}")
        audio.cue("error")
        audio.speak("Something went wrong describing the scene.", audio.PRIORITY_ALERT)
        return

    # "stop" arrived while we were waiting on the network: stay silent.
    if cancel_event.is_set():
        return

    print(f"[Describe] {text}")
    audio.cue("done")
    audio.speak(text, audio.PRIORITY_NORMAL)


if __name__ == "__main__":
    import threading

    describe_process(threading.Event())
    audio.get_speaker().wait_idle(60)
