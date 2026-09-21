import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Chưa cấu hình GEMINI_API_KEY trong file .env")

# Khởi tạo Gemini Client chính thức
client = genai.Client(api_key=GEMINI_API_KEY)