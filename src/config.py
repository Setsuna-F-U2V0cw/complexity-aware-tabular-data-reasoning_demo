import os

# Base Configuration
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TIMEOUT = int(os.getenv("TIMEOUT", "90"))

# Model Configuration Dictionary
# Define separate configuration for each agent
# Each agent needs:
# - model_name: The name of the model to use
# - api_key: The API key for the model service (default "EMPTY" for local VLLM)
# - base_url: The API endpoint URL
# In our experiments, the API model provider is the DashScope platform (https://dashscope.aliyuncs.com/compatible-mode/v1)

MODEL_CONFIG = {
    "retrieval": {
        "model_name": "${your model name}",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "planning": {
        "model_name": "${your model name}",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "coding": {
        "model_name": "${your model name}",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "debugging": {
        "model_name": "${your model name}",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
}

