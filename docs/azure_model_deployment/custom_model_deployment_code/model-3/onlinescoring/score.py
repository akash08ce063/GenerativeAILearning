import json
import os
import logging
import torch
from transformers import pipeline

# Define the global pipeline object
pipe = None


def init():
    """
    This function is called when the container is initialized.
    It loads the model and prepares it for inference.
    """
    global pipe
    try:
        # Use the model path defined by the environment variable AZUREML_MODEL_DIR
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "phi")

        # Check if the model path exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Initialize the Hugging Face pipeline
        pipe = pipeline(
            "text-generation", 
            model=model_path,  # Load the model from the path
            device="cpu",     # Use GPU (CUDA) for inference
            torch_dtype=torch.bfloat16  # Use bfloat16 for optimized performance on supported hardware
        )
        
        logging.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logging.error(f"Error initializing the pipeline: {e}")
        raise e


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to process the request.
    It takes the input data, performs inference, and returns the result.
    """
    try:
        # Parse the input data (expected as JSON)
        data = json.loads(raw_data)
        input_data = data.get("input_data")
        messages = input_data.get("messages")
        
        if not messages:
            raise ValueError("Input data is missing 'messages' field")
        
        # Log the received input for debugging
        logging.info(f"Received input: {messages}")
        
        # Pass the messages to the pipeline for text generation
        output = pipe(messages, max_new_tokens=50)
        
        # Return the output as a JSON response
        return json.dumps({"generated_text": output})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return json.dumps({"error": str(e)})