# Aria Glasses Qwen2.5VL Assistive AI

## Project Overview
- Real-time captioning and Q&A pipeline for visually impaired users using Meta Project Aria glasses
- Uses Qwen2.5VL vision-language model (via Ollama) for image captioning and follow-up Q&A
- Runs locally on macOS (Apple Silicon), designed for extensibility

## Repository Structure
- `aria_ai_caption/` → Main codebase (caption server, client, wake word, etc.)
- `aria_env/`     → Python virtual environment (not included in repo)
- `README.md`    → This file

## Requirements
- Meta Project Aria Gen 1 glasses and SDK (see Meta documentation)
- Python 3.11+ (recommend Apple Silicon for best performance)
- [Ollama](https://ollama.com/) (v0.7.0+) with Qwen2.5VL model pulled locally
- Android Debug Bridge (ADB) for USB communication
- All Python dependencies in `aria_ai_caption/requirements.txt`

> **Note:** Large model files (Qwen2.5VL, Whisper, etc.) are not included in this repo. Download them via Ollama or the appropriate tool.

## Setup Instructions

### 1. Clone the Repository
```sh
# (from your home or projects directory)
git clone https://github.com/SirAlchemist1/Aria-glasses-on-Qwen.git
cd "aria pipeline/aria_ai_caption"
```

### 2. Set Up Python Environment
```sh
python3 -m venv ../aria_env
source ../aria_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/openai/whisper.git
```

### 3. Install and Set Up Ollama
- Download and install Ollama from [ollama.com](https://ollama.com/)
- Pull the Qwen2.5VL model:
```sh
ollama pull qwen2.5vl:latest
```

### 4. Install Meta Project Aria SDK
- Download and install the SDK from Meta's official site
- Set `PYTHONPATH` if needed (see Meta documentation)

### 5. Connect Your Glasses
- Plug in your Aria glasses via USB
- Ensure ADB is installed (`brew install android-platform-tools`)
- Check device with `adb devices`

## Running the System

### Terminal 1: Start the Caption Server
```sh
cd "/Users/jarvis/aria pipeline/aria_ai_caption"
source ../aria_env/bin/activate
python caption_server.py
```

### Terminal 2: Start the Main Client
```sh
cd "/Users/jarvis/aria pipeline/aria_ai_caption"
source ../aria_env/bin/activate
python aria_server_caption.py --interface usb
```

- The system will wait for the wake word ("Hey Aria") before captioning.
- After the wake word, the next image will be captioned and spoken aloud.
- Only one caption per wake word (as of latest code).

## Usage Notes
- Press 't' to type a follow-up question
- Press 's' to speak a follow-up question
- Press 'q' to quit

## Troubleshooting
- If the video feed doesn't appear, ensure both terminals are running
- If captioning doesn't start after saying "Hey Aria", check that the caption server is running and the wake word is detected
- If you see connection errors, check ADB and USB connection
- If you get `ModuleNotFoundError: aria`, ensure the Meta Project Aria SDK is installed and in your `PYTHONPATH`

## Large Files
- **Model files (Qwen2.5VL, Whisper, etc.) are not included in this repo.**
- Download them using Ollama or the appropriate tool as described above.

---

For more details or to contribute, see the [GitHub repo](https://github.com/SirAlchemist1/Aria-glasses-on-Qwen).
