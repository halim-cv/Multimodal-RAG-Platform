import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

print(f"Testing model: {model_name}")
client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model=model_name,
        contents="Hello sibling! How are you doing today?"
    )
    print("Success! Response from sibling:")
    print(response.text)
except Exception as e:
    print(f"Error with model {model_name}: {e}")
    print("Available models:")
    for m in client.models.list():
        print(m.name)
