# This runs GPU server connection 
# Make sure this is running before running programs needing server requests.
# IMPORTANT: Change your server address 
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from ollama import Client

# === Initialize Flask and Ollama ===
app = Flask(__name__)
client = Client()  #connects to local Ollama server at http://localhost:11434
# can use "qwen2.5vl:latest" or other supported models
ai_model = f"qwen2.5vl:latest"

# === Prompts ===
CAPTION_PROMPT = (
    "You are a visual assistant for someone who is visually impaired. "
    "Describe the scene in detail, focusing on: "
    "1. Main objects and people in view (be specific about what you see) "
    "2. Their spatial relationships (left, right, center, distance) "
    "3. Any potential hazards or important details "
    "4. The overall context and setting "
    "Keep the description clear, concise, and under three sentences. "
    "Be specific about distances and directions when relevant. "
    "Do not start with 'I see' or 'I can see' - just describe what's there."
)

FOLLOW_UP_PROMPT_TEMPLATE = (
    "You are a visual assistant for someone who is visually impaired. "
    "Answer the following question about the image concisely and accurately: {}"
)

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

        #generate caption
        response = client.generate(
            model=ai_model,
            prompt=CAPTION_PROMPT,
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
        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === Run server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded = True)

