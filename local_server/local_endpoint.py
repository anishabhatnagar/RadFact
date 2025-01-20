from flask import Flask, request, jsonify

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging

from datetime import datetime

app = Flask(__name__)

# Set up the logger
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"/gpfs/scratch/ab10945/Imp_Generation/RadFact/local_server/endpoint_logs/endpoint_{current_time}.log"
logging.basicConfig(
    level=logging.INFO,  # Log all INFO and above messages
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # Save logs to a file
        logging.StreamHandler()              # Print logs to the console
    ]
)
logger = logging.getLogger(__name__)

# Specify the path to your local LLM
MODEL_PATH = "/gpfs/scratch/ab10945/Imp_Generation/Meta-Llama-3.1-8B-Instruct"  # Replace with the path to your local model

# Load the model and tokenizer on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map='auto')

@app.route('/completions', methods=['POST'])
def completions():
    try:
        # Log the incoming request
        # logger.info("Received request: %s", request.get_json())
        logger.info(f"Model device: {next(model.parameters()).device}")

        data = request.get_json()

        # Extract and validate the input
        # model = data.get("model", None)
        prompt = data.get("prompt", None)
        max_tokens = data.get("max_tokens", 1024)
        temperature = data.get("temperature", 0.0)
        do_sample=temperature > 0.0
        top_p = data.get("top_p", 1.0)
        frequency_penalty = data.get("frequency_penalty", 0.0)
        presence_penalty = data.get("presence_penalty", 0.0)
        stop = data.get("stop", None)
        num_return_sequences = data.get("n", 1)
        eos_token_id=tokenizer.eos_token_id if stop is None else tokenizer.convert_tokens_to_ids(stop)

        if not model or not prompt:
            logger.warning("Missing 'model' or 'prompt' in the request: %s", data)
            return jsonify({'error': "'model' and 'prompt' are required fields."}), 400

        # Log the extracted model and prompt
        # logger.info("Model: %s", model)
        # logger.info("Prompt: %s", prompt)

        # Example processing of the prompt (you can replace this with your model logic)
        # Simulate the response for testing
        responses = []


        for single_prompt in prompt:
            input = tokenizer(single_prompt, return_tensors='pt', truncation=True).to(device)
            logger.info(f"Input tensor device: {input['input_ids'].device}")

            output_ids = model.generate(
                input["input_ids"],
                attention_mask=input["attention_mask"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,  # Enable sampling if temperature > 0
                # frequency_penalty=frequency_penalty,
                # presence_penalty=presence_penalty,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

            output = output_ids[0][input["input_ids"].shape[-1]:]
            impression = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(impression)

        # Log the generated response
        logger.info("Generated response: %s", responses)

        # Return the response
        return jsonify({"responses": responses})

    except Exception as e:
        logger.exception("An error occurred while processing the request")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
