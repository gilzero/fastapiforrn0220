# filepath: models.py
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Literal
from dotenv import load_dotenv
import os

load_dotenv()

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 6000))
MAX_MESSAGES_IN_CONTEXT = int(os.getenv("MAX_MESSAGES_IN_CONTEXT", 50))

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[int] = None
    model: Optional[str] = None  # Kept for response tracking

    @field_validator('content')
    @classmethod
    def validate_content_length(cls, v):
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters")
        return v

class ChatRequest(BaseModel):
    messages: List[ConversationMessage]

    @field_validator('messages')
    @classmethod
    def validate_messages_count(cls, v):
        if len(v) > MAX_MESSAGES_IN_CONTEXT:
            raise ValueError(f"Conversation exceeds maximum of {MAX_MESSAGES_IN_CONTEXT} messages")
        return v

class HealthResponse(BaseModel):
    status: Literal["OK", "ERROR"]
    message: Optional[str] = None
    provider: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[Dict[str, str]] = None