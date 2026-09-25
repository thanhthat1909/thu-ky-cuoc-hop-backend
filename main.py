import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from schemas import ChatRequest, ChatResponse, MeetingSummaryResponse
from services import MeetingService


app = FastAPI(
    title="Thư ký cuộc họp API",
    version="2.0.0",
    description="Backend xử lý âm thanh cuộc họp bằng Gemini.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp_audio"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "Thư ký cuộc họp API đang hoạt động.",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/api/v1/meetings/process-audio",
    response_model=MeetingSummaryResponse,
)
async def process_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Thiếu tên file âm thanh.")

    mime_type = file.content_type or "audio/mp4"

    allowed_mimes = {
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/wav",
        "audio/x-wav",
        "audio/aac",
        "audio/flac",
        "audio/ogg",
        "audio/webm",
    }

    # Một số điện thoại gửi M4A với MIME application/octet-stream.
    # Không chặn trường hợp này; Gemini có thể tự nhận diện theo nội dung.
    if mime_type not in allowed_mimes and not mime_type.startswith("audio/"):
        mime_type = "audio/mp4"

    suffix = Path(file.filename).suffix.lower() or ".audio"
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    file_path = TEMP_DIR / safe_name

    try:
        with file_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        result = MeetingService.process_audio_meeting(
            file_path=str(file_path),
            mime_type=mime_type,
        )
        return result

    except RuntimeError as exc:
        # Giữ nguyên file ở điện thoại; backend chỉ xóa file tạm của Render.
        message = str(exc)

        # 429/503 đã được đổi thành thông báo thân thiện trong services.py.
        status_code = 429 if (
            "429" in message
            or "vượt giới hạn" in message
            or "hết hạn mức" in message
        ) else 503 if (
            "quá tải" in message
            or "không sẵn sàng" in message
        ) else 500

        raise HTTPException(status_code=status_code, detail=message)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi hệ thống: {exc}",
        )

    finally:
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass

        await file.close()


@app.post("/api/v1/meetings/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        answer = MeetingService.ask_meeting_assistant(
            transcript=request.transcript,
            question=request.question,
        )
        return ChatResponse(answer=answer)

    except RuntimeError as exc:
        message = str(exc)
        status_code = 429 if (
            "429" in message
            or "vượt giới hạn" in message
            or "hết hạn mức" in message
        ) else 500

        raise HTTPException(status_code=status_code, detail=message)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi hệ thống: {exc}",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
