# This runs GPU server connection 
# Make sure this is running before running programs needing server requests.
# IMPORTANT: Change your server address 
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from ollama import Client
import re

# === Initialize Flask and Ollama ===
app = Flask(__name__)
client = Client()  #connects to local Ollama server at http://localhost:11434
# can use "qwen2.5vl:latest" or other supported models
ai_model = f"qwen2.5vl:latest"

# === User Step Length (meters) ===
USER_STEP_LENGTH = 0.7  # Default average step length in meters; can be user-configurable

# === Prompts ===
CAPTION_PROMPT = (
    "You are an assistive visual guide for someone who is blind or visually impaired. "
    "Describe the scene simply and clearly, focusing on what would help the user understand and navigate. "
    "Include: "
    "1. Main objects, people, and their locations (left, right, center, near, far). "
    "2. Any text that is visible and readable. "
    "3. Hazards, obstacles, or anything blocking the way. "
    "4. Directions or distances to important features (like exits, signs, open doors). "
    "5. Lighting conditions if relevant. "
    "6. If a door is visible, estimate how far it is (in meters or feet). "
    "7. If stairs are visible, count the number of stair steps and mention if there is a railing. "
    "Be concise, avoid complex language, and do not use phrases like 'I see'. "
    "Keep the description under three sentences."
)

FOLLOW_UP_PROMPT_TEMPLATE = (
    "You are an assistive visual guide for someone who is blind or visually impaired. "
    "Answer the following question about the image simply and directly. "
    "If the question is about distance to an object (like a door), estimate the distance in meters or feet. "
    "If the question is about stairs, count the number of visible stair steps and mention if there is a railing. "
    "Give clear, step-by-step or spatial answers if needed. "
    "If the question is about text, read it out. "
    "Question: {}"
)

def build_prompt(image_path, lux=None, depth=None):
    context = []
    if lux is not None:
        if lux < 50:
            context.append("The scene is very dimly lit.")
        else:
            context.append("The scene is well-lit.")
    if depth is not None:
        context.append(f"The main subject is about {depth:.1f} metres away.")
    # Compose the context sentence
    context_hint = " ".join(context) if context else None
    # Existing prompt logic
    # ... existing code ...
    # When building the final prompt, prepend context_hint if present
    # For example:
    # prompt = f"{context_hint} {CAPTION_PROMPT}" if context_hint else CAPTION_PROMPT
    # ... existing code ...

def distance_to_steps(distance_m, step_length=USER_STEP_LENGTH):
    if distance_m is None or step_length <= 0:
        return None
    steps = int(round(distance_m / step_length))
    return steps if steps > 0 else 1

def extract_distance(text):
    # Look for patterns like 'X meters', 'X m', 'X feet', 'X ft'
    m = re.search(r"([0-9]+\.?[0-9]*)\s*(meters|meter|m)", text)
    if m:
        return float(m.group(1)), 'm'
    m = re.search(r"([0-9]+\.?[0-9]*)\s*(feet|foot|ft)", text)
    if m:
        # Convert feet to meters
        return float(m.group(1)) * 0.3048, 'ft'
    return None, None

def append_steps_to_caption(caption, step_length=USER_STEP_LENGTH):
    distance_m, unit = extract_distance(caption)
    if distance_m is not None:
        steps = distance_to_steps(distance_m, step_length)
        if steps:
            # Try to find the object (e.g., door) in the sentence
            m = re.search(r"(door|object|exit|stairs|sign)", caption, re.IGNORECASE)
            obj = m.group(1) if m else "object"
            return f"{caption} The nearest {obj} seems to be about {steps} steps away."
    return caption

@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        #load and preprocess image
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((512, 512))  #increased resolution for better detail

        #conversts to base64 png
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Optional context_hint
        context_hint = request.form.get('context_hint') or request.args.get('context_hint')

        #generate caption
        prompt = CAPTION_PROMPT
        if context_hint:
            prompt = f"{context_hint} {prompt}"

        response = client.generate(
            model=ai_model,
            prompt=prompt,
            images=[image_b64],
            options={
                "temperature": 0.7,  # Lower temperature for more focused responses
                "top_p": 0.9,       # Higher top_p for more diverse but relevant responses
                "num_predict": 100   # Limit response length
            }
        )

        caption = response.get("response", "No caption returned.")
        # Clean up the response
        caption = caption.strip()
        if caption.startswith("I see") or caption.startswith("I can see"):
            caption = caption[2:].strip()
        # --- Add step conversion logic ---
        caption = append_steps_to_caption(caption)
        return jsonify({'caption': caption})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/follow_up', methods=['POST'])
def follow_up():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'Missing image or question'}), 400

    try:
        image_file = request.files['image']
        question = request.form['question']

        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((256, 256))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        prompt = FOLLOW_UP_PROMPT_TEMPLATE.format(question)

        response = client.generate(
            model=ai_model,  # use "qwen2.5vl:latest"
            prompt=prompt,
            images=[image_b64],
        )

        answer = response.get("response", "No answer returned.")
        # --- Add step conversion logic ---
        answer = append_steps_to_caption(answer)
        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === Run server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded = True)

