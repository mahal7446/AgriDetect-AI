import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

# Get API key
api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key loaded: {api_key[:20]}...")

# Configure Gemini
genai.configure(api_key=api_key)

# Test with simple prompt
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Say hello in one sentence")

print(f"\nResponse: {response.text}")
print("\n✅ Gemini API is working!")
