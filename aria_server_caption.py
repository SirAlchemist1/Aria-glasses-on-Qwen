# Combination of all test code to run through server. Integrates llava captioning
# for aria glasses, text to speech, q&a, and wake word.
# q&a requests can be also triggered on terminal by pressing "t" or speech can be prompeted
# by pressing "s"

import cv2, argparse, time, io, threading, requests, queue, os, sys, warnings
import numpy as np
from PIL import Image
from ollama import Client
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
from datetime import datetime

#speech to text imports
import whisper
import sounddevice as sd
import tempfile, subprocess
import scipy.io.wavfile as wavfile

#Wake word inputs
import wake_word #make sure wake_word.py is in same folder

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# === Initialize Ollama client ===
print("Connecting to Ollama...")
client = Client()
print("Qwen2.5VL (Ollama) client initialized.")

# === Initialize Whisper ===
#Change to faster-whisper foe faster CPU transcription?
print("Loading Whisper model...")
stt_model = whisper.load_model("base", device = "cpu") # also can pick "tiny", "base", "small", and "medium"
print("Whisper loaded.")

#Variables for tts interruption
tts_queue = queue.Queue()
tts_lock  = threading.Lock()
current_tts_proc = None 

def record_audio(duration=4, fs=16000):
    print("Listening...")
    recording = sd.rec(int(duration * fs), samplerate=fs,
                    channels=1, dtype="int16")
    sd.wait()
    print("Recording finished.")
    return recording, fs

def transcribe_audio(recording, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, fs, recording)
        print("Transcribing...")
        result = stt_model.transcribe(f.name, fp16=False)
    os.remove(f.name)
    return result["text"].strip()

def stop_current_tts(): #interrupts speech and empty the queue
    global current_tts_proc
    with tts_lock:
        if current_tts_proc and current_tts_proc.poll() is None:
            current_tts_proc.terminate()   #or .kill() for immediate stop
        current_tts_proc = None
        #flush anything that was still in line
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
            except queue.Empty:
                break

def log_event(message, logfile="wake_log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]  {message}")
    #Uncomment following if you want datalog txt
    #with open(logfile, "a") as f:
    #    f.write(f"[{timestamp}] {message}\n")

# === Streaming Observer Class ===
class StreamingObserver:
    def __init__(self):
        self.last_image = None
        self.last_caption_time = 0
        self.cooldown = 10  # seconds between captions **CHANGE IF NEEDED**
        self.caption = "Waiting for image..."
        self.caption_in_progress = False
        self.last_caption = ""
        self.tts_in_progress = False #flag for determining tts in progess
        self.caption_pause = False #flag for pausing caption when answering question
        self.caption_enabled = False # Only allow captioning after wake word

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
        if record.camera_id == aria.CameraId.Rgb:
            self.last_image = np.rot90(image, -1)
            self.maybe_caption()

    def maybe_caption(self):
        now = time.time()
        if (
            self.last_image is not None
            and not self.caption_pause
            and not self.caption_in_progress
            and not self.tts_in_progress
            and now - self.last_caption_time >= self.cooldown
            and self.caption_enabled  # Only caption if enabled by wake word
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
            self.caption_enabled = False # Require wake word again for next caption

    def generate_caption(self, np_img: np.ndarray) -> str:
        try:
            #convert numpy image to PIL and encode as PNG in memory
            image = Image.fromarray(np_img).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            #send image to Flask caption server (make sure server is running)
            files = {'image': ('frame.png', buffer, 'image/png')}
            response = requests.post(
                "http://127.0.0.1:8000/caption",
                files=files,
                timeout=15  # seconds till timeout. adjust as needed
            )

            if response.status_code == 200:
                return response.json().get("caption", "No caption received.")
            else:
                return f"Server error: {response.status_code} - {response.text}"
        
        except requests.exceptions.ConnectTimeout:
            return "Caption server timed out; is it running?"
        
        except Exception as e:
            return f"Exception during captioning: {e}"

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

            response = requests.post("http://127.0.0.1:8000/follow_up", files=files, data=data, timeout=15)

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

# === Initialize text to speech queue and worker ===
def tts_worker():
    global current_tts_proc
    while True:
        text = tts_queue.get()
        if text is None:# shutdown signal
            break
        try:
            observer.tts_in_progress = True #flag on 
            # launch `say` non-blocking:
            cmd = ["say", text]
            with tts_lock:
                current_tts_proc = subprocess.Popen(cmd)
            current_tts_proc.wait()#block until done or killed
        except Exception as e:
            print("TTS error:", e)
        finally:
            with tts_lock:
                current_tts_proc = None
            observer.tts_in_progress = False #flag off 
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
            # ------- choose input mode -------
            mode = input("\n[t]ype or [s]peak a follow-up (exit = q): ").strip().lower()
            if mode == "q":
                print("Exiting follow-up loop.")
                sys.exit(0)

            if mode == "t":
                question = input("Your question: ").strip()
                if not question:
                    continue

            elif mode == "s":
                #pause captioning and stop tts
                observer.caption_pause = True
                stop_current_tts()
                time.sleep(0.1) #pause for speakers to stop

                #record & transcribe question
                audio, rate = record_audio(duration=4)
                question = transcribe_audio(audio, rate)
                print(f'You said: "{question}"')
                if not question:
                    observer.caption_pause = False
                    continue

                #ask Qwen2.5VL & speak the answer immediately
                answer = observer.ask_follow_up(question)
                print("\nQwen2.5VL says:", answer, flush=True)

                observer.tts_in_progress = True #block captions while we speak
                stop_current_tts() #nothing should slip ahead in the queue
                subprocess.call(["say", answer]) #synchronous, returns when done
                observer.tts_in_progress = False
                observer.caption_pause = False #resumes normal caption flow
                continue
            
            else:
                print("Enter t, s, or q.")
                continue
            # ------- send to Qwen2.5VL -------
            answer = observer.ask_follow_up(question)
            print("\nQwen2.5VL says:", answer, flush=True)

            # enqueue answer for speech after any caption
            while observer.caption_in_progress:
                time.sleep(0.05)
            tts_queue.put(answer)

    except (KeyboardInterrupt, EOFError):
        print("\nStopping follow-up loop.")
        stop_current_tts()
        tts_queue.put(None)
        sys.exit(0)

def follow_up_on_wake(observer: StreamingObserver):
    while True:
        wake_word.wait_for_wake_word(stt_model)
        log_event("Wake word detected.")

        observer.caption_pause = True
        stop_current_tts()
        time.sleep(0.1)

        # Do NOT re-enable captioning here (option 1: only one caption per run)
        # observer.caption_enabled = True

        print("Listening for your question...")
        audio, rate = record_audio(duration=4)
        question = transcribe_audio(audio, rate)
        print(f'You asked: "{question}"')
        log_event(f"Question: {question}")

        if question:
            answer = observer.ask_follow_up(question)
            print("\nQwen2.5VL says:", answer)
            log_event(f"Answer: {answer}")
            subprocess.call(["say", answer])

        observer.caption_pause = False
        time.sleep(1.0)  # prevent retriggering

wake_thread = threading.Thread(target=follow_up_on_wake, args=(observer,), daemon=True)
wake_thread.start()

try:
    while True:
        if observer.last_image is not None:
            frame = cv2.cvtColor(observer.last_image, cv2.COLOR_RGB2BGR)
            #display caption in window
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
        
        time.sleep(0.01)#sleep for 10ms

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    streaming_client.unsubscribe()
    stop_current_tts()   # Stops current tts
    tts_queue.put(None)  # Signal TTS thread to exit
    tts_thread.join()    # Wait for TTS thread to finish
    cv2.destroyAllWindows()
    print("Exiting.")