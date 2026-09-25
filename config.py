import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Chưa cấu hình GEMINI_API_KEY trên Render.")

# Có thể đổi model trên Render bằng biến môi trường GEMINI_MODEL.
# Giữ model cũ làm mặc định để không làm thay đổi hành vi hiện tại.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Số lần thử thêm khi Gemini trả lỗi tạm thời (429/503).
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

# Không cho backend chờ vô hạn vì một request HTTP từ điện thoại.
GEMINI_MAX_RETRY_WAIT_SECONDS = float(
    os.getenv("GEMINI_MAX_RETRY_WAIT_SECONDS", "35")
)

client = genai.Client(api_key=GEMINI_API_KEY)
