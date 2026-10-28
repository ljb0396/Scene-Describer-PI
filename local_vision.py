import os
import threading
from collections import OrderedDict

import cv2

_yolo = None
_lock = threading.Lock()

ALIASES = {"people": "person", "man": "person", "woman": "person", "phone": "cell phone",
           "table": "dining table", "couch": "sofa", "monitor": "tv", "computer": "laptop"}
IRREGULAR = {"person": "people", "knife": "knives", "sheep": "sheep", "scissors": "scissors"}


def _model():
    global _yolo
    with _lock:
        if _yolo is None:
            from ultralytics import YOLO
            _yolo = YOLO(os.environ.get("YOLO_WEIGHTS", "yolov8n.pt"))
        return _yolo


def preload():
    """Call once at startup in a background thread, so the first offline answer isn't slow."""
    import numpy as np
    _model()(np.zeros((320, 320, 3), np.uint8), imgsz=320, verbose=False)


def detect(frame, conf=0.4):
    m = _model()
    r = m(frame, imgsz=320, conf=conf, verbose=False)[0]
    h, w = frame.shape[:2]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append({"label": m.names[int(b.cls[0])],
                    "cx": (x1 + x2) / 2 / w,
                    "area": (x2 - x1) * (y2 - y1) / (w * h)})
    out.sort(key=lambda d: d["area"], reverse=True)       # biggest (nearest) first
    return out


def _where(d):
    pos = "on your left" if d["cx"] < 0.33 else "on your right" if d["cx"] > 0.67 else "ahead"
    return pos + (", close" if d["area"] > 0.25 else "")


def _phrase(label, n):
    if n == 1:
        return ("an " if label[0] in "aeiou" else "a ") + label
    return f"{n} {IRREGULAR.get(label, label + 's')}"


def describe(frame):
    dets = detect(frame)
    if not dets:
        return "I don't see anything I recognize."
    groups = OrderedDict()
    for d in dets:
        g = groups.setdefault(d["label"], {"n": 0, "first": d})
        g["n"] += 1
    items = []
    for label, g in list(groups.items())[:4]:
        text = _phrase(label, g["n"])
        items.append(text if g["n"] > 1 else f"{text} {_where(g['first'])}")
    joined = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return f"I see {joined}."


def read_text(frame, min_conf=50):
    try:
        import pytesseract
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(gray, config="--psm 6",
                                         output_type=pytesseract.Output.DICT)
    except Exception as e:                        # includes TesseractNotFoundError
        return "Text reading isn't available offline." if "tesseract" in str(e).lower() \
            else "I couldn't read the image."
    words = [w for w, c in zip(data["text"], data["conf"])
             if w.strip() and float(c) >= min_conf]
    return " ".join(words) or "I don't see any text."


def answer_question(question, frame):
    q = question.lower()
    if any(w in q for w in ("read", "say", "text", "sign", "label", "written", "write")):
        return read_text(frame)
    dets = detect(frame)
    names = {m for m in _model().names.values()}
    wanted = {ALIASES.get(w, w) for w in q.replace("?", "").split()} | \
             {n for n in names if n in q}
    wanted &= names
    if wanted:
        replies = []
        for label in wanted:
            hits = [d for d in dets if d["label"] == label]
            replies.append(f"Yes, I see {_phrase(label, 1)} {_where(hits[0])}." if hits
                           else f"I don't see {_phrase(label, 1)}.")
        return " ".join(replies)
    return "I can only list what I recognize offline. " + describe(frame)