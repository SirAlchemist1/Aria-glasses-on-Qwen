# Placeholder for llava_laptop.py
# TODO: Implement functionality as needed to match reference repo # Program for running llava captioning on laptop cpu/gpu. 
# Includes speech to text and follow up questions
import cv2
import numpy as np
import time
from PIL import Image
from ollama import Client
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
import argparse
import base64
import io
import threading
import requests
import queue
import os
import sys

# === Initialize Ollama client ===
print("Connecting to Ollama...")
client = Client()
print("Qwen2.5VL (Ollama) client initialized.")

# === Streaming Observer Class ===
class StreamingObserver:
    def __init__(self):
        self.last_image = None
        self.last_caption_time = 0
        self.cooldown = 1.5  # seconds between captions
        self.caption = "Waiting for image..."
        self.caption_in_progress = False
        self.last_caption = ""
        self.tts_in_progress = False

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
        if record.camera_id == aria.CameraId.Rgb:
            self.last_image = np.rot90(image, -1)
            self.maybe_caption()

    def maybe_caption(self):
        now = time.time()
        if (
            self.last_image is not None
            and not self.caption_in_progress
            and not self.tts_in_progress #ensures text to speech is complete
            and now - self.last_caption_time >= self.cooldown
        ):
            print("Triggering captioning...")
            self.caption_in_progress = True
            self.last_caption_time = now
            threading.Thread(
                target=self._caption_worker, args=(self.last_image.copy(),)
            ).start()

    def _caption_worker(self, image):
        try:
            start = time.time()
            caption = self.generate_caption(image)
            duration = time.time() - start

            self.caption = caption
            self.last_caption = caption
            print("Caption from Qwen2.5VL:", caption)
            print(f"Caption generation took {duration:.2f} seconds")
            tts_queue.put(caption) #speak the caption
        finally:
            self.caption_in_progress = False

    def generate_caption(self, np_img: np.ndarray) -> str:
        try:
            # Convert image to PIL and resize
            image = Image.fromarray(np_img).convert("RGB")
            image = image.resize((256, 256))

            # Encode to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Call Ollama Qwen2.5VL model
            response = client.generate(
                model="qwen2.5vl:latest", # use "qwen2.5vl:latest" or other supported Qwen2.5VL models
                prompt="Describe this image in a short, informative sentence for someone who is visually impaired.",
                images=[image_b64],
            )
            return response.get("response", "No caption returned.")
        except Exception as e:
            return f"Exception: {e}"
    
    def ask_follow_up(self, question: str) -> str:
        if self.last_image is None:
            return "No image available yet for follow-up."

        try:
            qa_start = time.time()
            image = Image.fromarray(self.last_image).convert("RGB").resize((256, 256))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            files = {'image': ('frame.png', buffer, 'image/png')}
            data = {'question': question}  # assumes your server accepts this

            response = requests.post("http://10.100.241.227:8000/follow_up", files=files, data=data)

            if response.status_code == 200:
                print(f"Q&A took {time.time() - qa_start:.2f} seconds")
                return response.json().get("answer", "No answer returned.")
            else:
                return f"Server error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Exception during follow-up: {e}"


# === CLI Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument(
    "--interface",
    type=str,
    required=True,
    choices=["usb", "wifi"],
    help="Connection type: usb or wifi",
)
args = parser.parse_args()

# === Optional WiFi Device Setup ===
if args.interface == "wifi":
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    device_client.set_client_config(client_config)
    device = device_client.connect()
    streaming_manager = device.streaming_manager

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = "profile18"
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config
    streaming_manager.start_streaming()
    print("Streaming started over Wi-Fi.")

# === Aria Streaming Setup ===
print("Initializing Aria streaming client...")
aria.set_log_level(aria.Level.Info)
streaming_client = aria.StreamingClient()

config = streaming_client.subscription_config
config.subscriber_data_type = aria.StreamingDataType.Rgb
config.message_queue_size[aria.StreamingDataType.Rgb] = 1

options = aria.StreamingSecurityOptions()
options.use_ephemeral_certs = True
config.security_options = options

streaming_client.subscription_config = config

# === Observer & Streaming Start ===
observer = StreamingObserver()

# === Initialize TTS queue and worker ===
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            observer.tts_in_progress = True #flag start
            os.system(f'say "{text}"')  # macOS built-in TTS
        except Exception as e:
            print("TTS Error:", e)
        finally:
            observer.tts_in_progress = False #flag end
            tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

streaming_client.set_streaming_client_observer(observer)
streaming_client.subscribe()
print("Connected to Aria. Streaming started.")

# === OpenCV Display Loop ===
cv2.namedWindow("Aria RGB + Qwen2.5VL Caption", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Aria RGB + Qwen2.5VL Caption", 640, 480)

# === Launch user input thread for follow-up questions ===
def follow_up_input_loop(observer: StreamingObserver):
    try:
        while True:
            question = input("\n Ask a follow-up question: ")
            if question.lower() == "exit":
                print("Exiting follow-up input thread.")
                sys.exit(0)
            answer = observer.ask_follow_up(question)
            print("\n Qwen2.5VL says:", answer, flush=True)
    except (KeyboardInterrupt, EOFError):
        print("\n Stopping follow-up loop.")
        sys.exit(0)

input_thread = threading.Thread(target=follow_up_input_loop, args=(observer,))
input_thread.daemon = True
input_thread.start()

try:
    while True:
        if observer.last_image is not None:
            frame = cv2.cvtColor(observer.last_image, cv2.COLOR_RGB2BGR)
            # Draw caption
            caption_display = observer.caption[:80] + "..." if len(observer.caption) > 80 else observer.caption
            cv2.putText(
                frame,
                caption_display,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Aria RGB + Qwen2.5VL Caption", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        time.sleep(0.01)  #sleep for 10ms to reduce cpu load

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    streaming_client.unsubscribe()
    tts_queue.put(None)  # Signal TTS thread to exit
    tts_thread.join()    # Wait for TTS thread to finish
    cv2.destroyAllWindows()
    print("Exiting.")