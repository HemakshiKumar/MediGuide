import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Model settings
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Hugging Face Token loaded strictly from environment variable
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
