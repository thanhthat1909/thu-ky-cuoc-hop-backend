import json
import os
import random
import re
import threading
import time
from typing import Any, Optional

from google import genai
from google.genai import types

from config import (
    GEMINI_MAX_RETRIES,
    GEMINI_MAX_RETRY_WAIT_SECONDS,
    GEMINI_MODEL,
    client,
)
from schemas import MeetingSummaryResponse


# Chỉ cho một yêu cầu Gemini xử lý audio chạy tại một thời điểm trên
# một instance Render. Điều này giúp tránh việc hai điện thoại gửi đồng thời
# và cùng đẩy project vượt RPM.
_GEMINI_LOCK = threading.Lock()


def _exception_text(exc: Exception) -> str:
    """Lấy thông tin lỗi an toàn từ exception của google-genai."""
    parts = [str(exc)]

    for attr in ("message", "details", "response"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(str(value))

    return " ".join(parts)


def _status_code(exc: Exception) -> Optional[int]:
    for attr in ("code", "status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    text = _exception_text(exc)
    match = re.search(r"\b(429|408|500|502|503|504)\b", text)
    return int(match.group(1)) if match else None


def _is_retryable(exc: Exception) -> bool:
    code = _status_code(exc)
    if code in (408, 429, 500, 502, 503, 504):
        return True

    text = _exception_text(exc).upper()
    return any(
        marker in text
        for marker in (
            "RESOURCE_EXHAUSTED",
            "TOO MANY REQUESTS",
            "RATE LIMIT",
            "UNAVAILABLE",
            "SERVICE UNAVAILABLE",
        )
    )


def _retry_after_seconds(exc: Exception, attempt: int) -> float:
    """
    Ưu tiên thời gian Gemini yêu cầu trong thông báo lỗi:
    'Please retry in 14.9729s'
    Nếu không có, dùng exponential backoff + jitter.
    """
    text = _exception_text(exc)

    patterns = (
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        r"retry_delay[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        r"seconds[^0-9]*([0-9]+(?:\.[0-9]+)?)",
    )

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return min(
                float(match.group(1)) + 1.0,
                GEMINI_MAX_RETRY_WAIT_SECONDS,
            )

    # 2, 4, 8... giây + jitter.
    backoff = min(2 ** attempt, GEMINI_MAX_RETRY_WAIT_SECONDS)
    jitter = random.uniform(0.2, 1.0)
    return min(backoff + jitter, GEMINI_MAX_RETRY_WAIT_SECONDS)


def _looks_like_daily_quota(text: str) -> bool:
    text = text.lower()
    return any(
        marker in text
        for marker in (
            "requests per day",
            "per day",
            "daily quota",
            "rpd",
            "quota will reset",
        )
    )


def _friendly_gemini_error(exc: Exception) -> RuntimeError:
    text = _exception_text(exc)
    code = _status_code(exc)

    if code == 429 or "RESOURCE_EXHAUSTED" in text.upper():
        if _looks_like_daily_quota(text):
            return RuntimeError(
                "Gemini đã hết hạn mức trong ngày của project. "
                "Vui lòng chờ quota reset hoặc nâng cấp billing/quota "
                "trong Google AI Studio. File ghi âm trên điện thoại vẫn được giữ."
            )

        return RuntimeError(
            "Gemini đang vượt giới hạn yêu cầu tạm thời (429). "
            "Hệ thống đã tự động thử lại nhưng quota vẫn chưa sẵn sàng. "
            "Vui lòng đợi khoảng 30–60 giây rồi gửi lại. "
            "File ghi âm gốc vẫn được giữ trên điện thoại."
        )

    if code in (503, 502, 504):
        return RuntimeError(
            "Gemini đang quá tải hoặc tạm thời không sẵn sàng. "
            "Hệ thống đã tự động thử lại. Vui lòng gửi lại sau ít phút."
        )

    return RuntimeError(f"Lỗi Gemini: {text}")


def _generate_with_retry(
    *,
    contents: Any,
    config: types.GenerateContentConfig,
) -> Any:
    """
    Gọi Gemini với retry có kiểm soát.
    Không retry các lỗi cấu hình/401/403/404/400.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_exc = exc

            if not _is_retryable(exc) or attempt >= GEMINI_MAX_RETRIES:
                raise _friendly_gemini_error(exc) from exc

            # Nếu đã biết quota là quota theo ngày thì retry cũng không giúp.
            if _looks_like_daily_quota(_exception_text(exc)):
                raise _friendly_gemini_error(exc) from exc

            wait_seconds = _retry_after_seconds(exc, attempt)
            time.sleep(wait_seconds)

    raise _friendly_gemini_error(last_exc)  # pragma: no cover


class MeetingService:
    PRIMARY_MODEL = GEMINI_MODEL

    @staticmethod
    def process_audio_meeting(file_path: str, mime_type: str) -> MeetingSummaryResponse:
        uploaded_file = None

        try:
            uploaded_file = client.files.upload(
                file=file_path,
                config=types.UploadFileConfig(mime_type=mime_type),
            )

            # Chờ Gemini xử lý file audio thành ACTIVE.
            max_wait = 180
            started = time.time()

            while getattr(uploaded_file.state, "name", "") == "PROCESSING":
                if time.time() - started > max_wait:
                    raise RuntimeError(
                        "Gemini xử lý file âm thanh quá lâu. "
                        "File gốc vẫn được giữ để bạn gửi lại."
                    )

                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)

            state_name = getattr(uploaded_file.state, "name", "")
            if state_name == "FAILED":
                raise RuntimeError(
                    "Gemini không thể đọc file âm thanh. "
                    "Hãy thử lại bằng file MP3, M4A hoặc WAV."
                )

            prompt = """
Bạn là thư ký cuộc họp tiếng Việt.

Hãy phân tích TOÀN BỘ file âm thanh cuộc họp và trả về ĐÚNG JSON theo schema được cung cấp.

Yêu cầu:
1. transcript:
   - Chép lại nội dung cuộc họp càng đầy đủ càng tốt.
   - Giữ nguyên tiếng Việt, tên người, số liệu và thuật ngữ khi nghe được.
   - Không tự bịa nội dung không có trong audio.

2. overview:
   - Tóm tắt ngắn gọn nội dung và mục đích chính của cuộc họp.

3. key_points:
   - Liệt kê các nội dung quan trọng, quyết định, vấn đề, kết luận.
   - Chỉ lấy thông tin có trong cuộc họp.

4. action_items:
   - Liệt kê công việc cần thực hiện sau cuộc họp.
   - task: công việc.
   - assignee: người/nhóm phụ trách nếu xác định được, nếu không ghi "Chưa chỉ định".
   - deadline: hạn hoàn thành nếu xác định được, nếu không ghi "Chưa rõ".

Nếu âm thanh không đủ rõ để xác định người hoặc thời hạn, không được đoán.
Trả về JSON hợp lệ, không thêm Markdown hay giải thích bên ngoài JSON.
"""

            response_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MeetingSummaryResponse,
                temperature=0.1,
            )

            # Serialize các request Gemini để giảm burst 429 trên cùng Render instance.
            with _GEMINI_LOCK:
                response = _generate_with_retry(
                    contents=[uploaded_file, prompt],
                    config=response_config,
                )

            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini trả về kết quả rỗng.")

            # Một số SDK/model có thể trả JSON kèm whitespace; pydantic xử lý được.
            return MeetingSummaryResponse.model_validate_json(text)

        except RuntimeError:
            raise
        except Exception as exc:
            raise _friendly_gemini_error(exc) from exc

        finally:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    # Không để lỗi xóa file tạm làm hỏng kết quả đã tạo.
                    pass

    @staticmethod
    def ask_meeting_assistant(transcript: str, question: str) -> str:
        if not transcript.strip():
            raise RuntimeError("Cuộc họp chưa có transcript để hỏi đáp.")

        prompt = f"""
Bạn là trợ lý cho một cuộc họp.

Chỉ trả lời dựa trên transcript bên dưới.
Nếu transcript không đủ thông tin để trả lời, hãy nói rõ "Không tìm thấy thông tin này trong transcript", không tự bịa.

TRANSCRIPT:
{transcript}

CÂU HỎI:
{question}
"""

        response_config = types.GenerateContentConfig(
            temperature=0.2,
        )

        try:
            with _GEMINI_LOCK:
                response = _generate_with_retry(
                    contents=prompt,
                    config=response_config,
                )

            answer = getattr(response, "text", None)
            if not answer:
                raise RuntimeError("Gemini trả về câu trả lời rỗng.")
            return answer.strip()

        except RuntimeError:
            raise
        except Exception as exc:
            raise _friendly_gemini_error(exc) from exc
