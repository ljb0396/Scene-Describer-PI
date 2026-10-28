import socket
import json
import time
import threading
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from test_tesseract import ocr_process
from test_yolo_gui import yolo_process

# --- Configuration ---
# NOTE: this must be a *local* address to bind to, not the ESP32's address/URL.
# "" or "0.0.0.0" means "listen on all network interfaces of this machine".
UDP_IP = "0.0.0.0"
UDP_PORT = 3333          # Match the port your ESP32 is sending to
SAMPLE_RATE = 16000      # 16kHz matches Vosk and your config
CHANNELS = 1             # Mono
DTYPE = 'int16'          # 2 bytes per sample

CHUNK_SIZE = 1024
BYTES_PER_SAMPLE = 2
BUFFER_SIZE = CHUNK_SIZE * BYTES_PER_SAMPLE * CHANNELS

# How long we can go without a packet before warning "no audio received"
NO_DATA_TIMEOUT = 3.0    # seconds
# Socket recv timeout, so the loop can periodically check for silence/timeouts
SOCKET_TIMEOUT = 0.1     # seconds
# RMS level below which we consider a packet "silent" (tune to your mic/gain)
SILENCE_RMS_THRESHOLD = 50

grammar = '["read", "see", "stop", "[unk]"]'


# Initialize UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(SOCKET_TIMEOUT)
print(f"Listening for UDP audio on port {UDP_PORT}...")


def rms_level(data: bytes) -> float:
    """Compute RMS amplitude of a chunk of int16 PCM audio."""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))

# A single shared stop flag for whatever task is currently active
cancel_current_task = threading.Event()
active_thread = None

def dispatch_new_task(task_function):
    global active_thread
    
    # 1. If something is already running, kill it first
    if active_thread and active_thread.is_alive():
        print("Switching commands! Signaling previous task to stop...")
        cancel_current_task.set() 
        active_thread.join()      # Wait for the old thread to fully clean up and exit
    
    # 2. Reset the flag and start the new task
    cancel_current_task.clear()
    active_thread = threading.Thread(target=task_function, args=(cancel_current_task,))
    active_thread.start()


def main():
    # 1. Load the Vosk Model
    try:
        print("Loading Vosk model... (This may take a few seconds)")
        model = Model("model")
        recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure your model folder is unzipped and named 'model' in this directory.")
        return

    print("\n>>> System active. Listening for ESP32 audio... (Ctrl+C to stop) <<<\n")

    # --- Reception tracking state ---
    packet_count = 0
    last_packet_time = None
    warned_no_data = False
    last_stats_time = time.time()

    try:

        while True:
	 

            # 3. Receive raw bytes from ESP32 over UDP (with timeout, so we can
            #    detect "no data" instead of blocking forever)
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                data = None

            now = time.time()

            if data:
                packet_count += 1
                last_packet_time = now
                warned_no_data = False

                level = rms_level(data)
                status = "SILENT" if level < SILENCE_RMS_THRESHOLD else "audio"

                # Lightweight per-packet confirmation, once a second-ish worth
                if packet_count % 20 == 0:
                    print(f"[RX] pkt#{packet_count} from {addr} | {len(data)} bytes | "
                          f"RMS={level:.0f} ({status})")

                # 5. Feed the exact same raw bytes into Vosk for Speech-to-Text
                if recognizer.AcceptWaveform(data):
                    result_json = json.loads(recognizer.Result())
                    text = result_json.get("text", "")
                    
                    print(f"\nFinal Text: {text}")

                    if text == "read":
                        dispatch_new_task(ocr_process)
                    
                    if text == "see":
                        dispatch_new_task(yolo_process)

                    if text == "stop":
                        cancel_current_task.set()

                else:
                    partial_json = json.loads(recognizer.PartialResult())
                    partial_text = partial_json.get("partial", "")
                    if partial_text:
                        print(f"Live: {partial_text}", end="\r", flush=True)
            else:
                # No packet arrived within SOCKET_TIMEOUT seconds
                if last_packet_time is None:
                    elapsed_since_start = now - last_stats_time
                    if elapsed_since_start > NO_DATA_TIMEOUT and not warned_no_data:
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
        sock.close()

if __name__ == "__main__":
    main()
