"""Shared video-source helpers so the live-preview scripts can pull frames from
either a local webcam or the ESP32-CAM MJPEG stream.

Selection is via the VIDEO_SOURCE env var:
  - unset / "0" / "1" ...      -> cv2.VideoCapture(<int>)  (local webcam)
  - "http://<esp-ip>/stream"   -> MJPEG stream off the ESP32-CAM

The ESP firmware (main/camera_streamer.c) serves a
"multipart/x-mixed-replace; boundary=frame" stream of JPEG frames at /stream.
"""

import os

import cv2
import numpy as np
import requests

DEFAULT_ESP_URL = "http://192.168.4.1/stream"

# The OV3660 module may be mounted inverted. VIDEO_ROTATE = 0/90/180/270
# (degrees clockwise) corrects it in software for every consumer.
_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def apply_rotation(frame):
    code = _ROTATE_CODES.get(int(os.environ.get("VIDEO_ROTATE", "0")))
    return frame if code is None else cv2.rotate(frame, code)


class MJPEGStream:
    """Minimal MJPEG reader with a cv2.VideoCapture-ish interface.

    Only the bits the preview scripts need: isOpened(), read(), release().
    """

    def __init__(self, url, timeout=5, chunk_size=4096):
        self.url = url
        self.timeout = timeout
        self.chunk_size = chunk_size
        self._resp = None
        self._chunks = None
        self._buf = bytearray()
        self._open()

    def _open(self):
        self._resp = requests.get(self.url, stream=True, timeout=self.timeout)
        self._resp.raise_for_status()
        self._chunks = self._resp.iter_content(chunk_size=self.chunk_size)

    def isOpened(self):  # noqa: N802 - match cv2.VideoCapture
        return self._resp is not None

    def read(self):
        while True:
            start = self._buf.find(b"\xff\xd8")
            end = self._buf.find(b"\xff\xd9", start + 2) if start != -1 else -1

            if start != -1 and end != -1:
                jpg = bytes(self._buf[start:end + 2])
                del self._buf[:end + 2]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                return True, apply_rotation(frame)

            try:
                chunk = next(self._chunks)
            except (StopIteration, requests.RequestException):
                return False, None
            if not chunk:
                return False, None
            self._buf.extend(chunk)
            # Guard against unbounded growth if we never see a full frame.
            if len(self._buf) > 4_000_000:
                del self._buf[:-1_000_000]

    def release(self):
        if self._resp is not None:
            self._resp.close()
            self._resp = None


def open_video_source(default=DEFAULT_ESP_URL):
    """Return an object with isOpened()/read()/release() for the configured source."""
    source = os.environ.get("VIDEO_SOURCE", default)

    if source is None:
        return cv2.VideoCapture(1) #Testing

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        print(f"Opening ESP32-CAM MJPEG stream: {source}")
        return MJPEGStream(source)

    try:
        index = int(source)
    except (TypeError, ValueError):
        raise SystemExit(f"Unrecognized VIDEO_SOURCE: {source!r}")
    return cv2.VideoCapture(index)
