# Created as an import module for server_llava_caption. Functionally same as hey_aria.py
# NEEDED as module for stt programs

import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import time
import scipy.io.wavfile as wavfile
from rapidfuzz import fuzz #allows for wider range of misheard wake words
import sys

model = whisper.load_model("base")

WAKE_PHRASES = ["hey aria", "hey area", "hey arya"]
THRESHOLD = 80 #threshold for wake word difference

def is_wake_phrase(text, phrases=WAKE_PHRASES, threshold=THRESHOLD):
    return any(fuzz.partial_ratio(text.lower(), phrase) >= threshold for phrase in phrases)

def record_chunk(duration=4, fs=16000): #records for 4 seconds, change duration for longer sample
    print("Listening for wake word...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio, fs

def transcribe(audio, fs, model):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, fs, audio)
        result = model.transcribe(f.name)
        os.remove(f.name)
        return result["text"]

def wait_for_wake_word(model):
    print("Waiting for 'Hey Aria' to start captioning...")
    try:
        while True:
            audio_chunk, rate = record_chunk()
            text = transcribe(audio_chunk, rate, model)
            print("Heard:", text)

            if is_wake_phrase(text):
                print("Wake phrase detected!")
                os.system('say "How can I help?"')
                break

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nWake-word detection interrupted.")
        sys.exit(0)