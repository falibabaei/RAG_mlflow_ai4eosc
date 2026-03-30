from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
mlflow_token = os.getenv("MLFLOW_TRACKING_PASSWORD")
print(f"Using API Key: {mlflow_token[:4]}...{mlflow_token[-4:]}")  # only partial display for security
client = OpenAI(
    base_url="https://mlflow-oidc.dev.ai4eosc.eu/gateway/openai/v1",
    api_key=API_KEY,

        default_headers={
        "Authorization": f"Bearer {mlflow_token}", 
    }
)

response = client.chat.completions.create(
    model="test",
    messages=[{"role": "user", "content": "How are you?"}]
)
print(response)