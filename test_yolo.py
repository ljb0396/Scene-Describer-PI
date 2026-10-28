import os
import cv2
import time
import threading
import audio_feedback as audio

from ultralytics import YOLO
from video_source import open_video_source


# ---------------------------------
# Configuration
# ---------------------------------

DIAGNOSTIC_FLOOR = 0.15
PRODUCTION_THRESHOLD = 0.7

# Cap inference rate so the Pi 4 CPU isn't pinned (it also needs to run
# Vosk). Raise on faster hardware.
MAX_FPS = float(os.environ.get("YOLO_MAX_FPS", "3"))


# ---------------------------------
# Load YOLO model
# ---------------------------------

model = YOLO(
    os.environ.get(
        'YOLO_WEIGHTS',
        'yolov8n.pt'
    )
)


# ---------------------------------
# Text-to-speech (shared, interruptible)
# ---------------------------------

def speak(text):

    audio.speak(text, audio.PRIORITY_LOW)


# ---------------------------------
# YOLO process
# ---------------------------------

def yolo_process(cancel_event):

    cap = None

    # Objects detected during the
    # previous YOLO inference
    previous_objects = set()

    # Holds ONLY the newest camera frame
    latest_frame = [None]

    # Protects latest_frame
    frame_lock = threading.Lock()

    try:

        # ---------------------------------
        # Connect to camera
        # ---------------------------------

        print("[YOLO] Connecting to camera...")

        cap = open_video_source()

        if not cap or not cap.isOpened():

            print(
                "[YOLO Error] "
                "Could not open video source."
            )

            return


        print("[YOLO] Camera connected.")


        # ---------------------------------
        # Camera reader thread
        # ---------------------------------

        def camera_reader():

            while not cancel_event.is_set():

                ok, frame = cap.read()

                if not ok:
                    # Don't spin the CPU when the stream drops
                    time.sleep(0.05)
                    continue

                # Always replace the old frame.
                # We NEVER allow a queue of old frames.
                with frame_lock:

                    latest_frame[0] = frame


        camera_thread = threading.Thread(
            target=camera_reader,
            daemon=True
        )

        camera_thread.start()


        print(
            "[YOLO] Object detection started."
        )


        # ---------------------------------
        # YOLO processing loop
        # ---------------------------------

        while not cancel_event.is_set():

            # Get the newest available frame
            with frame_lock:

                if latest_frame[0] is None:

                    frame = None

                else:

                    frame = latest_frame[0].copy()


            # Camera hasn't provided a frame yet
            if frame is None:

                time.sleep(0.01)
                continue


            # Check for stop command
            if cancel_event.is_set():
                break

            loop_started = time.monotonic()

            # ---------------------------------
            # Run YOLO on newest frame
            # ---------------------------------

            results = model(
                frame,
                verbose=False,
                conf=DIAGNOSTIC_FLOOR,
                device="cpu"
            )


            # ---------------------------------
            # Find detected objects
            # ---------------------------------

            current_objects = set()


            for box in results[0].boxes:

                score = float(box.conf[0])


                # Ignore low-confidence detections
                if score < PRODUCTION_THRESHOLD:
                    continue


                class_id = int(
                    box.cls[0]
                )


                label = model.names[
                    class_id
                ]


                current_objects.add(
                    label
                )


            # ---------------------------------
            # Find newly appearing objects
            # ---------------------------------

            newly_appeared = (
                current_objects
                - previous_objects
            )


            # ---------------------------------
            # Announce new objects
            # ---------------------------------

            for label in newly_appeared:

                if cancel_event.is_set():
                    break

                print(
                    f"[YOLO] New object: {label}"
                )

                speak(label)


            # Remember objects from this frame
            previous_objects = current_objects

            # Throttle to MAX_FPS
            spare = (1.0 / MAX_FPS) - (time.monotonic() - loop_started)
            if spare > 0:
                cancel_event.wait(spare)


        print(
            "[YOLO] Stop command received."
        )


    except Exception as e:

        print(
            f"[YOLO Error]: {e}"
        )


    finally:

        # ---------------------------------
        # Cleanup
        # ---------------------------------

        if cap is not None:

            try:

                cap.release()

                print(
                    "[YOLO] Network stream closed."
                )

            except Exception:
                pass


        cv2.destroyAllWindows()


        print(
            "[YOLO] Detection stopped."
        )
