import os
import cv2
import time
import threading
import pyttsx3

from ultralytics import YOLO
from video_source import open_video_source


# ---------------------------------
# Configuration
# ---------------------------------

DIAGNOSTIC_FLOOR = 0.15
PRODUCTION_THRESHOLD = 0.7

WINDOW_NAME = "ESP32 Live Object Detection"


# ---------------------------------
# Load YOLO model
# ---------------------------------

model = YOLO(
    os.environ.get(
        "YOLO_WEIGHTS",
        "yolov8n.pt"
    )
)


# ---------------------------------
# Text-to-speech
# ---------------------------------

engine = pyttsx3.init()

engine.setProperty(
    "rate",
    160
)

engine.setProperty(
    "volume",
    1.0
)


def speak(text):

    print(f"[YOLO] Speaking: {text}")

    engine.say(text)
    engine.runAndWait()


# ---------------------------------
# Draw detections
# ---------------------------------

def draw_detections(frame, result):

    for box in result.boxes:

        score = float(box.conf[0])

        class_id = int(box.cls[0])

        label = model.names[class_id]

        # Only display production-confidence detections
        if score < PRODUCTION_THRESHOLD:
            continue

        # Bounding box coordinates
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        text = f"{label} {score:.2f}"

        # Get text size
        (text_width, text_height), baseline = (
            cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
        )

        # Label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            (0, 255, 0),
            -1
        )

        # Label text
        cv2.putText(
            frame,
            text,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame


# ---------------------------------
# YOLO process
# ---------------------------------

def yolo_process(cancel_event):

    cap = None

    # Objects detected during previous inference
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
                    continue

                # Replace the previous frame.
                # This prevents a backlog of old frames.
                with frame_lock:

                    latest_frame[0] = frame


        camera_thread = threading.Thread(
            target=camera_reader,
            daemon=True
        )

        camera_thread.start()


        # ---------------------------------
        # Create GUI
        # ---------------------------------

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            800,
            600
        )


        print(
            "[YOLO] Object detection started."
        )

        print(
            "[YOLO] Press Q to close the GUI."
        )


        # ---------------------------------
        # YOLO processing loop
        # ---------------------------------

        while not cancel_event.is_set():

            # ---------------------------------
            # Get newest frame
            # ---------------------------------

            with frame_lock:

                if latest_frame[0] is None:

                    frame = None

                else:

                    frame = latest_frame[0].copy()


            # Camera hasn't produced a frame yet
            if frame is None:

                time.sleep(0.01)
                continue


            # ---------------------------------
            # Run YOLO
            # ---------------------------------

            results = model(
                frame,
                verbose=False,
                conf=DIAGNOSTIC_FLOOR,
                device="cpu"
            )

            result = results[0]


            # ---------------------------------
            # Draw detections
            # ---------------------------------

            annotated_frame = draw_detections(
                frame,
                result
            )


            # ---------------------------------
            # Find current objects
            # ---------------------------------

            current_objects = set()


            for box in result.boxes:

                score = float(box.conf[0])

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


            previous_objects = current_objects


            # ---------------------------------
            # Add status information
            # ---------------------------------

            cv2.putText(
                annotated_frame,
                "YOLO Object Detection",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # ---------------------------------
            # Show frame
            # ---------------------------------

            cv2.imshow(
                WINDOW_NAME,
                annotated_frame
            )


            # ---------------------------------
            # GUI keyboard handling
            # ---------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                print(
                    "[YOLO] Q pressed. Stopping..."
                )

                cancel_event.set()

                break


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


        cv2.destroyWindow(
            WINDOW_NAME
        )

        cv2.destroyAllWindows()


        print(
            "[YOLO] Detection stopped."
        )
