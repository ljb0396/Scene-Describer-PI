"""Live camera preview window for debugging from a dev machine.

The ESP32-CAM lives on its own Wi-Fi (192.168.4.1) that only the Pi joins,
so tunnel through the Pi first:

    ssh -N -L 8081:192.168.4.1:80 nebby@lance.local
    VIDEO_SOURCE=http://localhost:8081/stream VIDEO_ROTATE=180 python view_camera.py

Keys: q / Esc quit, s save snapshot to ./snapshots/

Note: the ESP32 may serve only one stream client at a time, so close this
before running a task on the Pi that opens the camera.
"""

import os
import time

import cv2

from video_source import open_video_source

WINDOW = "ESP32 camera (q to quit, s to save)"


def main():
    cap = open_video_source()
    if not cap or not cap.isOpened():
        raise SystemExit("Could not open video source.")

    frames = 0
    shown_fps = 0.0
    window_start = time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream ended or dropped.")
            break

        frames += 1
        now = time.monotonic()
        if now - window_start >= 1.0:
            shown_fps = frames / (now - window_start)
            frames = 0
            window_start = now

        h, w = frame.shape[:2]
        label = f"{w}x{h}  {shown_fps:.1f} fps"
        cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            os.makedirs("snapshots", exist_ok=True)
            path = os.path.join("snapshots", time.strftime("%Y%m%d-%H%M%S.jpg"))
            cv2.imwrite(path, frame)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
