import json
import socket
import threading
import time

import numpy as np
from vosk import KaldiRecognizer, Model

from tts import speak, stop_speaking
from vision_tasks import make_task

# --- Configuration ---
UDP_IP = "0.0.0.0"
UDP_PORT = 3333           # Match the port your ESP32 is sending to
SAMPLE_RATE = 16000
BUFFER_SIZE = 4096        # larger than the ESP32's packets, so nothing is truncated

NO_DATA_TIMEOUT = 3.0
SOCKET_TIMEOUT = 0.1
SILENCE_RMS_THRESHOLD = 50

QUESTION_TIMEOUT_S = 8.0  # how long to wait for the spoken question
QUESTION_MUTE_S = 1.2     # ignore audio while "Go ahead." is playing

grammar = '["read", "see", "question", "stop", "[unk]"]'

# Initialize UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(SOCKET_TIMEOUT)
print(f"Listening for UDP audio on port {UDP_PORT}...")


def rms_level(data: bytes) -> float:
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0


# A single shared stop flag for whatever task is currently active
cancel_current_task = threading.Event()
active_thread = None
_switch_lock = threading.Lock()


def _switch(task_function):
    global active_thread
    with _switch_lock:
        if active_thread and active_thread.is_alive():
            print("Switching commands! Signaling previous task to stop...")
            cancel_current_task.set()
            active_thread.join()
        cancel_current_task.clear()
        active_thread = threading.Thread(
            target=task_function, args=(cancel_current_task,), daemon=True)
        active_thread.start()


def dispatch_new_task(task_function):
    """Non-blocking: the switch happens in a helper thread, so audio keeps flowing."""
    threading.Thread(target=_switch, args=(task_function,), daemon=True).start()


def main():
    try:
        print("Loading Vosk model... (This may take a few seconds)")
        model = Model("model")
        recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure your model folder is unzipped and named 'model' in this directory.")
        return

    # Optional: warm up the offline YOLO model in the background
    def _preload():
        try:
            import local_vision
            local_vision.preload()
        except Exception as e:
            print(f"[preload] offline model not ready: {e}")
    threading.Thread(target=_preload, daemon=True).start()

    print("\n>>> System active. Listening for ESP32 audio... (Ctrl+C to stop) <<<\n")

    packet_count = 0
    last_packet_time = None
    warned_no_data = False
    last_stats_time = time.time()

    question_rec = None       # open-vocabulary recognizer, only alive during a question
    listen_from = 0.0
    question_deadline = 0.0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                data = None

            now = time.time()

            # Give up on a question if nothing usable arrived in time
            if question_rec is not None and now > question_deadline:
                print("\n[question] nothing heard")
                speak("I didn't catch that.", force=True)
                question_rec = None
                recognizer.Reset()

            if data:
                packet_count += 1
                last_packet_time = now
                warned_no_data = False

                level = rms_level(data)
                status = "SILENT" if level < SILENCE_RMS_THRESHOLD else "audio"
                if packet_count % 20 == 0:
                    print(f"[RX] pkt#{packet_count} from {addr} | {len(data)} bytes | "
                          f"RMS={level:.0f} ({status})")

                # --- Question mode: transcribe free speech instead of commands ---
                if question_rec is not None:
                    if now < listen_from:
                        continue
                    if question_rec.AcceptWaveform(data):
                        q = json.loads(question_rec.Result()).get("text", "").strip()
                        if q:
                            print(f"\n[question] {q}")
                            dispatch_new_task(make_task("ask", q))
                            question_rec = None
                            recognizer.Reset()
                    continue

                # --- Normal mode: fixed-grammar commands ---
                if recognizer.AcceptWaveform(data):
                    text = json.loads(recognizer.Result()).get("text", "")
                    print(f"\nFinal Text: {text}")

                    if text == "read":
                        dispatch_new_task(make_task("read"))
                    elif text == "see":
                        dispatch_new_task(make_task("see"))
                    elif text == "question":
                        cancel_current_task.set()
                        stop_speaking()
                        question_rec = KaldiRecognizer(model, SAMPLE_RATE)  # no grammar
                        listen_from = now + QUESTION_MUTE_S
                        question_deadline = listen_from + QUESTION_TIMEOUT_S
                        speak("Go ahead.", force=True)
                    elif text == "stop":
                        cancel_current_task.set()
                        stop_speaking()
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "")
                    if partial:
                        print(f"Live: {partial}", end="\r", flush=True)
            else:
                if last_packet_time is None:
                    if now - last_stats_time > NO_DATA_TIMEOUT and not warned_no_data:
                        print(f"\n[WARN] No audio received yet from any ESP32. "
                              f"Check IP/port/network (listening on 0.0.0.0:{UDP_PORT}).")
                        warned_no_data = True
                elif (now - last_packet_time) > NO_DATA_TIMEOUT and not warned_no_data:
                    print(f"\n[WARN] No audio received for {NO_DATA_TIMEOUT:.0f}s "
                          f"(last packet at {time.strftime('%H:%M:%S', time.localtime(last_packet_time))}). "
                          f"ESP32 may have stopped streaming.")
                    warned_no_data = True

    except KeyboardInterrupt:
        print("\nStopping audio detection...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        cancel_current_task.set()
        stop_speaking()
        if active_thread and active_thread.is_alive():
            active_thread.join(timeout=5)
        sock.close()


if __name__ == "__main__":
    main()
