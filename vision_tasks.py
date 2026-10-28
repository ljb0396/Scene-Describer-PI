import logging

import vision_router
from tts import speak
from video_source import open_video_source

log = logging.getLogger("vision")


def grab_frame(cap, timeout=5):
    """LatestFrameStream takes a timeout; MJPEGStream and cv2.VideoCapture don't."""
    try:
        return cap.read(timeout=timeout)
    except TypeError:
        return cap.read()


def make_task(kind, question=None):
    def run(cancel):
        cap = None
        try:
            try:
                cap = open_video_source()
                ok, frame = grab_frame(cap)
            except Exception as e:                  # ESP32 unreachable, bad URL, timeout...
                log.warning("camera error: %s", e)
                ok, frame = False, None
            if not ok or frame is None:
                speak("I can't get video right now.", force=True)
                return

            if kind == "see":
                text, source = vision_router.describe(frame)
            elif kind == "read":
                text, source = vision_router.read(frame)
            else:
                text, source = vision_router.ask(frame, question)

            if not cancel.is_set():
                log.info("[%s via %s] %s", kind, source, text)
                speak(text, force=True)
        except Exception:
            log.exception("task '%s' failed", kind)
            speak("Something went wrong.", force=True)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return run