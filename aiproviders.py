# filepath: aiproviders.py
from typing import List, Tuple
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
from models import ChatRequest, ConversationMessage
from logging_config import logger, debug_with_context

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
GENERIC_SYSTEM_PROMPT = os.getenv("GENERIC_SYSTEM_PROMPT",
                                "You are a helpful AI assistant that provides accurate and informative responses.")
GPT_SYSTEM_PROMPT = os.getenv("GPT_SYSTEM_PROMPT",
                              "You are ChatGPT, a helpful AI assistant that provides accurate and informative responses.")
CLAUDE_SYSTEM_PROMPT = os.getenv("CLAUDE_SYSTEM_PROMPT",
                                 "You are Claude, a highly capable AI assistant created by Anthropic, focused on providing accurate, nuanced, and helpful responses.")
GEMINI_SYSTEM_PROMPT = os.getenv("GEMINI_SYSTEM_PROMPT",
                                 "You are Gemini, a helpful and capable AI assistant created by Google, focused on providing clear and accurate responses.")

# Provider to model mapping
PROVIDER_MODELS = {
    "gpt": {"default": OPENAI_MODEL_DEFAULT, "fallback": OPENAI_MODEL_FALLBACK},
    "claude": {"default": ANTHROPIC_MODEL_DEFAULT, "fallback": ANTHROPIC_MODEL_FALLBACK},
    "gemini": {"default": GEMINI_MODEL_DEFAULT, "fallback": GEMINI_MODEL_FALLBACK}
}

# Add after loading environment variables
SUPPORTED_PROVIDERS = os.getenv("SUPPORTED_PROVIDERS", "gpt,claude,gemini").split(",")

# Validate supported providers
for provider in SUPPORTED_PROVIDERS:
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Environment Error: Provider '{provider}' in SUPPORTED_PROVIDERS not found in PROVIDER_MODELS")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
genai_client = genai.Client(api_key=GEMINI_API_KEY)


def get_system_prompt(messages: List[ConversationMessage]) -> str:
    """Get system prompt from messages or return generic default."""
    system_messages = [msg.content for msg in messages if msg.role == "system"]
    return " ".join(system_messages) if system_messages else GENERIC_SYSTEM_PROMPT


def get_provider_model(provider: str, use_fallback: bool = False) -> str:
    """Get the appropriate model for a provider."""
    if provider not in PROVIDER_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
    
    model_key = "fallback" if use_fallback else "default"
    return PROVIDER_MODELS[provider][model_key]


async def stream_response(request: ChatRequest, provider: str):
    message_id = f"{provider}-{int(time.time() * 1000)}"
    stream_start = time.time()
    chunks_sent = 0
    
    async def try_stream_with_model(model: str):
        """Attempt to stream response with given model."""
        try:
            system_prompt = get_system_prompt(request.messages)
            debug_with_context(logger, 
                f"Starting stream with model {model}",
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                message_count=len(request.messages)
            )

            if provider == "gpt":
                messages = [{"role": "system", "content": system_prompt}] + [
                    {"role": m.role, "content": m.content}
                    for m in request.messages if m.role != "system"
                ]
                debug_with_context(logger,
                    "Creating OpenAI stream",
                    model=model,
                    temperature=float(OPENAI_TEMPERATURE or 0.7),
                    max_tokens=int(OPENAI_MAX_TOKENS or 2000)
                )
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=float(OPENAI_TEMPERATURE or 0.7),
                    max_tokens=int(OPENAI_MAX_TOKENS or 2000),
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        chunks_sent += 1
                        data = {
                            "id": message_id,
                            "delta": {
                                "content": chunk.choices[0].delta.content,
                                "model": model
                            }
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
                stream_duration = time.time() - stream_start
                debug_with_context(logger,
                    f"Stream completed for {provider}",
                    duration=f"{stream_duration:.3f}s",
                    chunks_sent=chunks_sent,
                    model=model
                )

            elif provider == "claude":
                messages = [
                    {"role": m.role, "content": m.content}
                    for m in request.messages if m.role != "system"
                ]
                with anthropic_client.messages.stream(
                        model=model,
                        messages=messages,
                        system=CLAUDE_SYSTEM_PROMPT,
                        max_tokens=int(ANTHROPIC_MAX_TOKENS or 2000),
                        temperature=float(ANTHROPIC_TEMPERATURE or 0.7),
                ) as stream:
                    for text in stream.text_stream:
                        data = {
                            "id": message_id,
                            "delta": {
                                "content": text,
                                "model": model
                            }
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            elif provider == "gemini":
                config = types.GenerateContentConfig(
                    temperature=float(GEMINI_TEMPERATURE or 0.7),
                    max_output_tokens=int(GEMINI_MAX_TOKENS or 2000),
                    system_instruction=GEMINI_SYSTEM_PROMPT
                )
                chat_messages = [msg.content for msg in request.messages if msg.role != "system"]
                response = genai_client.models.generate_content_stream(
                    model=model,
                    contents=chat_messages,
                    config=config
                )
                for chunk in response:
                    if chunk.text:
                        data = {
                            "id": message_id,
                            "delta": {
                                "content": chunk.text,
                                "model": model
                            }
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error with {provider} using model {model}: {str(e)}", exc_info=True)
            return

    try:
        # Try with default model
        default_model = get_provider_model(provider)
        logger.info(f"Attempting to use default model {default_model} for {provider}")
        
        async for chunk in try_stream_with_model(default_model):
            yield chunk
            if chunk == "data: [DONE]\n\n":
                return

        # If we reach here, try fallback model
        fallback_model = get_provider_model(provider, use_fallback=True)
        logger.warning(f"Default model failed, attempting fallback model {fallback_model} for {provider}")
        
        async for chunk in try_stream_with_model(fallback_model):
            yield chunk
            if chunk == "data: [DONE]\n\n":
                return

        # If both models fail
        raise HTTPException(
            status_code=500,
            detail=f"Both default and fallback models failed for provider {provider}"
        )

    except Exception as e:
        logger.error(f"Stream response error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))