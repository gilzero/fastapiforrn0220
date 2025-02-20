# main.py
# filepath: pyserver/main.py
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
import asyncio
import time
import json
from datetime import datetime, UTC
import os
from typing_extensions import Literal
from dotenv import load_dotenv
import logging
import logging.handlers
from pathlib import Path
import sys
import psutil
import threading
import traceback
from openai import OpenAI  # Add this import at the top
from anthropic import Anthropic  # Add this import at the top
from google import genai  # Add this import at the top
from google.genai import types  # Add this import at the top

# load env variables
load_dotenv()

# load env variables from .env.example

PORT=os.getenv("PORT")

# API Keys
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# Model
OPENAI_MODEL_DEFAULT=os.getenv("OPENAI_MODEL_DEFAULT")
OPENAI_MODEL_FALLBACK=os.getenv("OPENAI_MODEL_FALLBACK")
ANTHROPIC_MODEL_DEFAULT=os.getenv("ANTHROPIC_MODEL_DEFAULT")
ANTHROPIC_MODEL_FALLBACK=os.getenv("ANTHROPIC_MODEL_FALLBACK")
GEMINI_MODEL_DEFAULT=os.getenv("GEMINI_MODEL_DEFAULT")
GEMINI_MODEL_FALLBACK=os.getenv("GEMINI_MODEL_FALLBACK")
GROQ_MODEL_DEFAULT=os.getenv("GROQ_MODEL_DEFAULT")
GROQ_MODEL_FALLBACK=os.getenv("GROQ_MODEL_FALLBACK")

# Temperature
OPENAI_TEMPERATURE=os.getenv("OPENAI_TEMPERATURE")
ANTHROPIC_TEMPERATURE=os.getenv("ANTHROPIC_TEMPERATURE")
GEMINI_TEMPERATURE=os.getenv("GEMINI_TEMPERATURE")
GROQ_TEMPERATURE=os.getenv("GROQ_TEMPERATURE")

# MAX TOKENS
OPENAI_MAX_TOKENS=os.getenv("OPENAI_MAX_TOKENS")        
ANTHROPIC_MAX_TOKENS=os.getenv("ANTHROPIC_MAX_TOKENS")
GEMINI_MAX_TOKENS=os.getenv("GEMINI_MAX_TOKENS")
GROQ_MAX_TOKENS=os.getenv("GROQ_MAX_TOKENS")

# Message Validation
# MAX_MESSAGE_LENGTH: Maximum characters per message (approximately 6000 tokens)
# MAX_MESSAGES_IN_CONTEXT: Maximum total messages in conversation (including both user & AI messages)
# MIN_MESSAGE_LENGTH: Minimum characters required for a message
MAX_MESSAGE_LENGTH=os.getenv("MAX_MESSAGE_LENGTH")
MAX_MESSAGES_IN_CONTEXT=os.getenv("MAX_MESSAGES_IN_CONTEXT")
MIN_MESSAGE_LENGTH=os.getenv("MIN_MESSAGE_LENGTH")

# Rate Limiting
RATE_LIMIT_WINDOW_MS=os.getenv("RATE_LIMIT_WINDOW_MS")
RATE_LIMIT_MAX_REQUESTS=os.getenv("RATE_LIMIT_MAX_REQUESTS")
GPT_RATE_LIMIT_MAX=os.getenv("GPT_RATE_LIMIT_MAX")
CLAUDE_RATE_LIMIT_MAX=os.getenv("CLAUDE_RATE_LIMIT_MAX")
GEMINI_RATE_LIMIT_MAX=os.getenv("GEMINI_RATE_LIMIT_MAX")
GROQ_RATE_LIMIT_MAX=os.getenv("GROQ_RATE_LIMIT_MAX")

# System Prompts for each provider
GPT_SYSTEM_PROMPT = os.getenv("GPT_SYSTEM_PROMPT", 
    "You are ChatGPT, a helpful AI assistant that provides accurate and informative responses.")

CLAUDE_SYSTEM_PROMPT = os.getenv("CLAUDE_SYSTEM_PROMPT", 
    "You are Claude, a highly capable AI assistant created by Anthropic, focused on providing accurate, nuanced, and helpful responses.")

GEMINI_SYSTEM_PROMPT = os.getenv("GEMINI_SYSTEM_PROMPT", 
    "You are Gemini, a helpful and capable AI assistant created by Google, focused on providing clear and accurate responses.")

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
def _setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Format for our log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File Handler - General logs
    file_handler = logging.handlers.RotatingFileHandler(
        "logs/app.log",
        maxBytes=10485760,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # File Handler - Error logs
    error_file_handler = logging.handlers.RotatingFileHandler(
        "logs/error.log",
        maxBytes=10485760,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    error_file_handler.setFormatter(formatter)
    error_file_handler.setLevel(logging.ERROR)
    logger.addHandler(error_file_handler)

    return logger

# Add after the existing setup_logging function
def _setup_debug_logging():
    debug_logger = logging.getLogger('debug')
    debug_logger.setLevel(logging.DEBUG)
    
    # Format for debug messages - more detailed than regular logs
    debug_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s\n'
        'File: %(pathname)s:%(lineno)d\n'
        'Function: %(funcName)s\n'
        'Message: %(message)s\n'
    )

    # File Handler - Debug logs
    debug_handler = logging.handlers.RotatingFileHandler(
        "logs/debug.log",
        maxBytes=20971520,  # 20MB
        backupCount=10,
        encoding="utf-8"
    )
    debug_handler.setFormatter(debug_formatter)
    debug_handler.setLevel(logging.DEBUG)
    debug_logger.addHandler(debug_handler)
    
    return debug_logger

# Initialize loggers
logger = _setup_logging()
debug_logger = _setup_debug_logging()

# Add this after the logging setup and before the route handlers
def _log_debug_info(
    action: str, 
    data: Any = None, 
    context: Dict[str, Any] = None, 
    error: Exception = None
):
    """Log detailed debug information"""
    debug_info = {
        "action": action,
        "timestamp": datetime.now(UTC).isoformat(),
        "data": data,
    }
    if error:
        debug_info["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }
    debug_logger.debug(f"Debug Info: {json.dumps(debug_info, default=str)}")

# Add after _log_debug_info function
def _format_log_data(data: Any) -> str:
    """Format data for logging in a readable way."""
    try:
        return json.dumps(data, indent=2, default=str)
    except:
        return str(data)

# Models
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[int] = None
    model: Optional[str] = None

    @field_validator('content')
    @classmethod
    def validate_content_length(cls, v):
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        if len(v) > int(MAX_MESSAGE_LENGTH):
            raise ValueError(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters")
        return v

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str

    @field_validator('messages')
    @classmethod
    def validate_messages_count(cls, v):
        if len(v) > int(MAX_MESSAGES_IN_CONTEXT):
            raise ValueError(f"Conversation exceeds maximum of {MAX_MESSAGES_IN_CONTEXT} messages")
        return v

class HealthResponse(BaseModel):
    status: Literal["OK", "ERROR"]
    message: Optional[str] = None
    provider: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[Dict[str, str]] = None

# Constants
MODELS = {
    "gpt": {
        "default": OPENAI_MODEL_DEFAULT,
        "fallback": OPENAI_MODEL_FALLBACK,
    },
    "claude": {
        "default": ANTHROPIC_MODEL_DEFAULT,
        "fallback": ANTHROPIC_MODEL_FALLBACK,
    },
    "gemini": {
        "default": GEMINI_MODEL_DEFAULT,
        "fallback": GEMINI_MODEL_FALLBACK,
    }
}

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Anthropic client
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Initialize Gemini client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check endpoint called")
    return {
        "status": "OK",
        "message": "System operational"
    }

@app.get("/health/{provider}", response_model=HealthResponse)
async def provider_health_check(provider: str):
    logger.info(f"Provider health check called for: {provider}")
    if provider not in MODELS:
        logger.error(f"Invalid provider requested: {provider}")
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
    
    start_time = time.time()
    await asyncio.sleep(0.1)
    duration = time.time() - start_time
    
    logger.info(f"Provider {provider} health check completed in {duration:.2f}s")
    return {
        "status": "OK",
        "provider": provider,
        "metrics": {"responseTime": duration}
    }

# Modify the _get_llm function to be a simple simulator
async def simulate_stream_response(message: str) -> str:
    """Simulate an AI response by returning words one at a time"""
    words = f"This is a simulated response for: {message}. The real API integration will be implemented later. This is just a placeholder response to show streaming functionality.".split()
    for word in words:
        yield word + " "
        await asyncio.sleep(0.1)  # Simulate network delay

# Add helper function before stream_response
def get_system_prompt(messages: List[ChatMessage]) -> str:
    """Get system prompt from messages or use default"""
    system_messages = [msg.content for msg in messages if msg.role == "system"]
    return " ".join(system_messages) if system_messages else GPT_SYSTEM_PROMPT

# Update the stream_response function
async def stream_response(request: ChatRequest, provider: str):
    message_id = f"{provider}-{int(time.time()*1000)}"
    start_time = time.time()
    
    try:
        # Get unified system prompt
        system_prompt = get_system_prompt(request.messages)
        
        if provider == "gpt":
            messages = [{"role": "system", "content": GPT_SYSTEM_PROMPT}] + [
                {"role": m.role, "content": m.content} 
                for m in request.messages if m.role != "system"
            ]
            stream = client.chat.completions.create(
                model=request.model,
                messages=messages,
                stream=True,
                temperature=float(OPENAI_TEMPERATURE or 0.7),
                max_tokens=int(OPENAI_MAX_TOKENS or 2000),
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    data = {
                        "id": message_id,
                        "delta": {"content": chunk.choices[0].delta.content}
                    }
                    yield f"data: {json.dumps(data)}\n\n"
        elif provider == "claude":
            messages = [
                {"role": m.role, "content": m.content}
                for m in request.messages if m.role != "system"
            ]
            with anthropic_client.messages.stream(
                model=request.model,
                messages=messages,
                system=CLAUDE_SYSTEM_PROMPT,  # Use Claude-specific prompt
                max_tokens=int(ANTHROPIC_MAX_TOKENS or 2000),
                temperature=float(ANTHROPIC_TEMPERATURE or 0.7),
            ) as stream:
                for text in stream.text_stream:
                    data = {
                        "id": message_id,
                        "delta": {"content": text}
                    }
                    yield f"data: {json.dumps(data)}\n\n"
        elif provider == "gemini":
            try:
                config = types.GenerateContentConfig(
                    temperature=float(GEMINI_TEMPERATURE or 0.7),
                    max_output_tokens=int(GEMINI_MAX_TOKENS or 2000),
                    system_instruction=GEMINI_SYSTEM_PROMPT  # Use Gemini-specific prompt
                )

                # Extract all non-system messages in order
                chat_messages = []
                for msg in request.messages:
                    if msg.role != "system":
                        chat_messages.append(msg.content)

                # Use streaming content generation
                response = genai_client.models.generate_content_stream(
                    model=request.model,
                    contents=chat_messages,  # Pass the conversation history
                    config=config
                )

                # Stream the response
                for chunk in response:
                    if chunk.text:
                        data = {
                            "id": message_id,
                            "delta": {"content": chunk.text}
                        }
                        yield f"data: {json.dumps(data)}\n\n"

            except Exception as e:
                logger.error(f"Gemini error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")
        else:
            # Keep simulation for other providers
            async for token in simulate_stream_response(request.messages[-1].content):
                yield f"data: {json.dumps({'id': message_id, 'delta': {'content': token}})}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        _log_debug_info(
            "stream_error",
            {
                "message_id": message_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "provider": provider
            },
            error=e
        )
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoints
@app.post("/chat/{provider}")
async def chat(provider: str, request: ChatRequest):
    start_time = time.time()
    
    try:
        # Validate provider
        if provider not in MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
        
        # Validate model
        valid_models = [MODELS[provider]["default"], MODELS[provider]["fallback"]]
        if request.model not in valid_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model for {provider}. Valid models are: {', '.join(valid_models)}"
            )
        
        # Log request metadata
        metadata = {
            "provider": provider,
            "model": request.model,
            "message_count": len(request.messages),
            "total_input_length": sum(len(msg.content) for msg in request.messages),
            "timestamp": datetime.now(UTC).isoformat()
        }
        logger.info(f"Chat request metadata:\n{_format_log_data(metadata)}")
        
        # Set up SSE response
        response = StreamingResponse(
            stream_response(request, provider),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
        # Log request duration
        duration = time.time() - start_time
        logger.info(f"Chat request initialized in {duration:.2f}s")
        
        return response
        
    except Exception as e:
        error_data = {
            "provider": provider,
            "model": request.model if hasattr(request, 'model') else None,
            "error": str(e),
            "error_type": type(e).__name__,
            "duration": time.time() - start_time
        }
        logger.error(f"Chat endpoint error:\n{_format_log_data(error_data)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat error with {provider}: {str(e)}"
        )

# Rate limiting middleware
@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    # Add rate limiting headers
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = RATE_LIMIT_MAX_REQUESTS
    response.headers["X-RateLimit-Remaining"] = str(int(RATE_LIMIT_MAX_REQUESTS) - 1)
    response.headers["X-RateLimit-Reset"] = str(int(time.time() + (int(RATE_LIMIT_WINDOW_MS)/1000)))
    return response

# Error handling
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_message = str(exc)
    if isinstance(exc, HTTPException):
        error_message = exc.detail
        logger.warning(f"HTTP Exception: {error_message}")
    else:
        logger.error(f"Unhandled exception: {error_message}", exc_info=True)
    
    if request.url.path.startswith("/chat/"):
        # Return error in SSE format for chat endpoints
        return StreamingResponse(
            iter([f'data: {{"error": "{error_message}"}}\ndata: [DONE]\n\n']),
            media_type="text/event-stream"
        )
    
    # Regular error response for other endpoints
    return {
        "status": "ERROR",
        "message": error_message,
        "error": {
            "type": exc.__class__.__name__,
            "message": error_message
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(PORT, "3050")  # Default to 3050 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)