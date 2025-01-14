from flask import Flask, request, jsonify
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the Flask app
app = Flask(__name__)

# Specify the path to your local LLM
MODEL_PATH = "/gpfs/scratch/ab10945/Imp_Generation/Meta-Llama-3.1-8B-Instruct"  # Replace with the path to your local model

# Load the model and tokenizer on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')

@app.route('/completions', methods=['POST'])
def completions():
    """
    Handle inference requests.
    Expects a JSON payload with 'input' key for input text.
    """
    try:
        # Handle raw JSON input without requiring a specific key
        data = request.get_json()
        if isinstance(data, dict):
            # If input is a dictionary, extract it (fallback to the `input` key if exists)
            input_text = data.get('input', '') or json.dumps(data)
        elif isinstance(data, str):
            # If input is a raw string, use it as is
            input_text = data
        else:
            return jsonify({'error': 'Invalid input format.'}), 400

        # Ensure input_text is valid
        if not input_text.strip():
            return jsonify({'error': 'Input text must be a non-empty string.'}), 400

        # Process the input_text
        # Replace this with the actual logic for the model
        response_text = f"Processed input: {input_text}"

        return jsonify({'response': response_text})

    except Exception as e:
        # Handle any errors during processing
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on all available network interfaces (default port: 5000)
    app.run(host='0.0.0.0', port=5000)
