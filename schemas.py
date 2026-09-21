from pydantic import BaseModel, Field
from typing import List, Optional

class ActionItem(BaseModel):
    task: str = Field(description="Nội dung công việc cần thực hiện")
    assignee: Optional[str] = Field(default="Chưa chỉ định", description="Người chịu trách nhiệm")
    deadline: Optional[str] = Field(default="Chưa rõ", description="Hạn chót hoàn thành")

class MeetingSummaryResponse(BaseModel):
    title: str = Field(description="Tiêu đề gợi ý cho cuộc họp")
    overview: str = Field(description="Tóm tắt tổng quan nội dung cuộc họp")
    key_points: List[str] = Field(description="Các điểm chính/quyết định quan trọng")
    action_items: List[ActionItem] = Field(description="Danh sách các việc cần làm (Action Items)")
    transcript: str = Field(description="Toàn bộ nội dung gỡ băng/phiên âm chi tiết từ file âm thanh")

class ChatRequest(BaseModel):
    transcript: str = Field(..., description="Văn bản bối cảnh cuộc họp")
    question: str = Field(..., description="Câu hỏi người dùng đặt ra cho cuộc họp")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời từ trợ lý AI")