import os
import time
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError
from config import client
from schemas import MeetingSummaryResponse

class MeetingService:
    # Model chuẩn duy nhất được Google cấp phép cho tài khoản của bạn
    PRIMARY_MODEL = "gemini-3.6-flash"

    @staticmethod
    def _get_exact_audio_mime_type(file_path: str, fallback_mime: str) -> str:
        """
        Xác định MIME type âm thanh chuẩn dựa vào đuôi mở rộng của file.
        """
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".mp3": "audio/mp3",
            ".m4a": "audio/m4a",
            ".wav": "audio/wav",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg"
        }
        return mime_map.get(ext, fallback_mime)

    @staticmethod
    async def process_audio_meeting(file_path: str, mime_type: str) -> MeetingSummaryResponse:
        """
        Upload file âm thanh lên Gemini File API và thực hiện tóm tắt cuộc họp.
        """
        # 1. Xác định MIME type chuẩn
        exact_mime = MeetingService._get_exact_audio_mime_type(file_path, mime_type)

        # 2. Upload file âm thanh lên server Gemini
        uploaded_file = client.files.upload(
            file=file_path,
            config=types.UploadFileConfig(mime_type=exact_mime)
        )

        # 3. Vòng lặp chờ Gemini xử lý file (ACTIVE)
        while True:
            file_info = client.files.get(name=uploaded_file.name)
            state_str = str(file_info.state)
            
            if "ACTIVE" in state_str:
                break
            elif "FAILED" in state_str:
                raise Exception("Gemini File API gặp lỗi trong quá trình xử lý file âm thanh.")
            
            time.sleep(2)

        # 4. Yêu cầu cho AI Thư ký
        prompt = """
        Bạn là một Thư ký cuộc họp chuyên nghiệp. Hãy lắng nghe file âm thanh cuộc họp này và thực hiện các nhiệm vụ sau:
        1. Gỡ băng chính xác toàn bộ nội dung cuộc họp (Transcript).
        2. Tóm tắt tổng quan nội dung cuộc họp.
        3. Rút ra các quyết định chính và điểm quan trọng.
        4. Trích xuất danh sách các việc cần làm (Action Items), bao gồm người thực hiện và hạn chót (nếu có).
        Response phải tuân theo chính xác định dạng JSON schema được yêu cầu.
        """

        response_text = None
        last_exception = None

        try:
            print(f"--> Đang gửi yêu cầu phân tích tới model: {MeetingService.PRIMARY_MODEL}")
            
            # Thử tối đa 3 lần nếu gặp lỗi quá tải tạm thời (503)
            for attempt in range(1, 4):
                try:
                    response = client.models.generate_content(
                        model=MeetingService.PRIMARY_MODEL,
                        contents=[file_info, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=MeetingSummaryResponse,
                            temperature=0.2,
                        ),
                    )
                    response_text = response.text
                    print(f"==> Thành công xử lý cuộc họp!")
                    break
                except ServerError as e:
                    last_exception = e
                    if e.code == 503:
                        print(f"⚠️ Server đang quá tải (503). Thử lại lần {attempt}/3 sau 3 giây...")
                        time.sleep(3)
                    else:
                        break
                except APIError as e:
                    last_exception = e
                    print(f"⚠️ Lỗi API: {e.message}")
                    break

            if not response_text:
                raise Exception(f"Không thể xử lý cuộc họp. Lỗi từ Gemini API: {last_exception}")

        finally:
            # 5. Luôn dọn dẹp file tạm trên Cloud Gemini
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                print(f"Lỗi dọn dẹp file tạm Gemini: {e}")

        # 6. Parse và validate dữ liệu trả về theo Schema
        return MeetingSummaryResponse.model_validate_json(response_text)

    @staticmethod
    async def ask_meeting_assistant(transcript: str, question: str) -> str:
        """
        Hỏi đáp dựa trên biên bản cuộc họp.
        """
        prompt = f"""
        Bạn là trợ lý cuộc họp "Thư ký cuộc họp". Dưới đây là biên bản cuộc họp:
        ---
        {transcript}
        ---
        Dựa vào biên bản cuộc họp trên, hãy trả lời câu hỏi sau của người dùng một cách chính xác và ngắn gọn:
        Câu hỏi: {question}
        """

        response = client.models.generate_content(
            model=MeetingService.PRIMARY_MODEL,
            contents=prompt,
        )
        return response.text