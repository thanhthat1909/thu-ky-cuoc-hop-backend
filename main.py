import os
import shutil
import traceback
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from schemas import MeetingSummaryResponse, ChatRequest, ChatResponse
from services import MeetingService

app = FastAPI(
    title="Thư ký cuộc họp API",
    description="Backend ứng dụng ghi chú, phiên âm và tóm tắt cuộc họp tự động.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Thư ký cuộc họp API đang hoạt động. Truy cập /docs để xem tài liệu API."}

@app.post(
    "/api/v1/meetings/process-audio",
    response_model=MeetingSummaryResponse,
    summary="Xử lý ghi âm/file âm thanh cuộc họp",
    status_code=status.HTTP_200_OK
)
async def process_audio(file: UploadFile = File(...)):
    file_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        # Lưu file vào thư mục tạm
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Gọi Service xử lý
        result = await MeetingService.process_audio_meeting(
            file_path=file_path,
            mime_type=file.content_type or "audio/mp3"
        )
        return result

    except Exception as e:
        # In chi tiết lỗi ra màn hình Terminal để kiểm tra
        print("=== LỖI XỬ LÝ AUDIO ===")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống: {str(e)}"
        )
    finally:
        # Xóa file tạm ở máy local
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post(
    "/api/v1/meetings/chat",
    response_model=ChatResponse,
    summary="Hỏi đáp thông minh với trợ lý về cuộc họp"
)
async def chat_with_meeting(payload: ChatRequest):
    try:
        answer = await MeetingService.ask_meeting_assistant(
            transcript=payload.transcript,
            question=payload.question
        )
        return ChatResponse(answer=answer)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi truy vấn trợ lý AI: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)