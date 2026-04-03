import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

with open("models.txt", "w", encoding="utf-8") as f:
    f.write("Available text embedding models:\n")
    for m in genai.list_models():
        f.write(f"{m.name}\n")

