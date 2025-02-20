# filepath: pyserver/aiproviders.py
from typing import List
import json
import time
import asyncio
from datetime import datetime, UTC
from dotenv import load_dotenv
import os
from fastapi import HTTPException
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
from models import ChatRequest, ChatMessage
from logging_config import logger

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configurations
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL_DEFAULT")
OPENAI_MODEL_FALLBACK = os.getenv("OPENAI_MODEL_FALLBACK")
ANTHROPIC_MODEL_DEFAULT = os.getenv("ANTHROPIC_MODEL_DEFAULT")
ANTHROPIC_MODEL_FALLBACK = os.getenv("ANTHROPIC_MODEL_FALLBACK")
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL_DEFAULT")
GEMINI_MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK")

# Temperature and max tokens
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE")
ANTHROPIC_TEMPERATURE = os.getenv("ANTHROPIC_TEMPERATURE")
GEMINI_TEMPERATURE = os.getenv("GEMINI_TEMPERATURE")
OPENAI_MAX_TOKENS = os.getenv("OPENAI_MAX_TOKENS")
ANTHROPIC_MAX_TOKENS = os.getenv("ANTHROPIC_MAX_TOKENS")
GEMINI_MAX_TOKENS = os.getenv("GEMINI_MAX_TOKENS")

# System Prompts
GPT_SYSTEM_PROMPT = os.getenv("GPT_SYSTEM_PROMPT",
                              "You are ChatGPT, a helpful AI assistant that provides accurate and informative responses.")
CLAUDE_SYSTEM_PROMPT = os.getenv("CLAUDE_SYSTEM_PROMPT",
                                 "You are Claude, a highly capable AI assistant created by Anthropic, focused on providing accurate, nuanced, and helpful responses.")
GEMINI_SYSTEM_PROMPT = os.getenv("GEMINI_SYSTEM_PROMPT",
                                 "You are Gemini, a helpful and capable AI assistant created by Google, focused on providing clear and accurate responses.")

MODELS = {
    "gpt": {"default": OPENAI_MODEL_DEFAULT, "fallback": OPENAI_MODEL_FALLBACK},
    "claude": {"default": ANTHROPIC_MODEL_DEFAULT, "fallback": ANTHROPIC_MODEL_FALLBACK},
    "gemini": {"default": GEMINI_MODEL_DEFAULT, "fallback": GEMINI_MODEL_FALLBACK}
}

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
genai_client = genai.Client(api_key=GEMINI_API_KEY)


def get_system_prompt(messages: List[ChatMessage]) -> str:
    system_messages = [msg.content for msg in messages if msg.role == "system"]
    return " ".join(system_messages) if system_messages else GPT_SYSTEM_PROMPT


async def stream_response(request: ChatRequest, provider: str):
    message_id = f"{provider}-{int(time.time() * 1000)}"

    try:
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
                    data = {"id": message_id, "delta": {"content": chunk.choices[0].delta.content}}
                    yield f"data: {json.dumps(data)}\n\n"

        elif provider == "claude":
            messages = [
                {"role": m.role, "content": m.content}
                for m in request.messages if m.role != "system"
            ]
            with anthropic_client.messages.stream(
                    model=request.model,
                    messages=messages,
                    system=CLAUDE_SYSTEM_PROMPT,
                    max_tokens=int(ANTHROPIC_MAX_TOKENS or 2000),
                    temperature=float(ANTHROPIC_TEMPERATURE or 0.7),
            ) as stream:
                for text in stream.text_stream:
                    data = {"id": message_id, "delta": {"content": text}}
                    yield f"data: {json.dumps(data)}\n\n"

        elif provider == "gemini":
            config = types.GenerateContentConfig(
                temperature=float(GEMINI_TEMPERATURE or 0.7),
                max_output_tokens=int(GEMINI_MAX_TOKENS or 2000),
                system_instruction=GEMINI_SYSTEM_PROMPT
            )
            chat_messages = [msg.content for msg in request.messages if msg.role != "system"]
            response = genai_client.models.generate_content_stream(
                model=request.model,
                contents=chat_messages,
                config=config
            )
            for chunk in response:
                if chunk.text:
                    data = {"id": message_id, "delta": {"content": chunk.text}}
                    yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream response error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))