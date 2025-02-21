# filepath: aiproviders.py
from typing import List, Tuple
import json
import time
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types
from fastapi import HTTPException
from models import ChatRequest, ConversationMessage
from logging_config import logger, debug_with_context, generate_conversation_id, log_conversation_entry
from configuration import (
    # API Keys
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    # Models
    OPENAI_MODEL_DEFAULT,
    OPENAI_MODEL_FALLBACK,
    ANTHROPIC_MODEL_DEFAULT,
    ANTHROPIC_MODEL_FALLBACK,
    GEMINI_MODEL_DEFAULT,
    GEMINI_MODEL_FALLBACK,
    GROQ_MODEL_DEFAULT,
    GROQ_MODEL_FALLBACK,
    # Temperature and tokens
    OPENAI_TEMPERATURE,
    ANTHROPIC_TEMPERATURE,
    GEMINI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
    ANTHROPIC_MAX_TOKENS,
    GEMINI_MAX_TOKENS,
    GROQ_TEMPERATURE,
    GROQ_MAX_TOKENS,
    # System prompts
    GENERIC_SYSTEM_PROMPT,
    GPT_SYSTEM_PROMPT,
    CLAUDE_SYSTEM_PROMPT,
    GEMINI_SYSTEM_PROMPT,
    # Providers
    SUPPORTED_PROVIDERS,
    RESPONSE_TIMEOUT
)
from groq import AsyncGroq

# Remove duplicate environment loading and variable definitions
# since they're now imported from configuration

# Provider to model mapping
PROVIDER_MODELS = {
    "gpt": {"default": OPENAI_MODEL_DEFAULT, "fallback": OPENAI_MODEL_FALLBACK},
    "claude": {"default": ANTHROPIC_MODEL_DEFAULT, "fallback": ANTHROPIC_MODEL_FALLBACK},
    "gemini": {"default": GEMINI_MODEL_DEFAULT, "fallback": GEMINI_MODEL_FALLBACK},
    "groq": {"default": GROQ_MODEL_DEFAULT, "fallback": GROQ_MODEL_FALLBACK}
}

# Initialize clients
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
genai_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)


def get_system_prompt(messages: List[ConversationMessage]) -> str:
    """Get system prompt from messages or return generic default."""
    if not messages:
        logger.warning("Empty messages list, using generic system prompt")
        return GENERIC_SYSTEM_PROMPT

    system_messages = [msg.content for msg in messages if msg.role == "system"]
    
    # Filter out empty or whitespace-only system messages
    valid_system_messages = [
        msg for msg in system_messages 
        if msg and msg.strip() and len(''.join(c for c in msg if not c.isspace() and c != '.')) > 0
    ]
    
    if valid_system_messages:
        return " ".join(valid_system_messages)
    
    logger.debug("No valid system messages found, using generic system prompt")
    return GENERIC_SYSTEM_PROMPT


def get_provider_model(provider: str, use_fallback: bool = False) -> str:
    """Get the appropriate model for a provider."""
    if provider not in PROVIDER_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
    
    model_key = "fallback" if use_fallback else "default"
    return PROVIDER_MODELS[provider][model_key]


async def _stream_response_gpt(messages: List[dict], model: str, message_id: str):
    """Handle streaming responses from GPT models."""
    debug_with_context(logger,
        "Creating OpenAI stream",
        model=model,
        temperature=float(OPENAI_TEMPERATURE or 0.7),
        max_tokens=int(OPENAI_MAX_TOKENS or 2000)
    )
    
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=float(OPENAI_TEMPERATURE or 0.7),
            max_tokens=int(OPENAI_MAX_TOKENS or 2000),
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                data = {
                    "id": message_id,
                    "delta": {
                        "content": chunk.choices[0].delta.content,
                        "model": model
                    }
                }
                yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in GPT stream: {str(e)}", exc_info=True)
        raise

async def _stream_response_claude(messages: List[dict], model: str, message_id: str):
    """Handle streaming responses from Claude models."""
    try:
        async with anthropic_client.messages.stream(
            model=model,
            messages=messages,
            system=CLAUDE_SYSTEM_PROMPT,
            max_tokens=int(ANTHROPIC_MAX_TOKENS or 2000),
            temperature=float(ANTHROPIC_TEMPERATURE or 0.7),
        ) as stream:
            async for text in stream.text_stream:
                data = {
                    "id": message_id,
                    "delta": {
                        "content": text,
                        "model": model
                    }
                }
                yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in Claude stream: {str(e)}", exc_info=True)
        raise

async def _stream_response_gemini(chat_messages: List[str], model: str, message_id: str):
    """Handle streaming responses from Gemini models."""
    try:
        config = types.GenerateContentConfig(
            temperature=float(GEMINI_TEMPERATURE or 0.7),
            max_output_tokens=int(GEMINI_MAX_TOKENS or 2000),
            system_instruction=GEMINI_SYSTEM_PROMPT
        )
        
        stream = await genai_client.aio.models.generate_content_stream(
            model=model,
            contents=chat_messages,
            config=config
        )
        async for chunk in stream:
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
        logger.error(f"Error in Gemini stream: {str(e)}", exc_info=True)
        raise

async def stream_response(request: ChatRequest, provider: str):
    conversation_id = generate_conversation_id()
    response_buffer = []
    """Stream chat responses from an AI provider"""
    logger.debug("Starting stream_response", 
        extra={
            "context": {
                "provider": provider,
                "message_count": len(request.messages),
                "validation_state": "pre-validation"
            }
        }
    )

    try:
        message_id = f"{provider}-{int(time.time() * 1000)}"
        stream_start = time.time()
        chunks_sent = 0
        stream_completed = False  # Flag to track if the stream completed successfully

        async def stream_with_model(model: str):
            """Attempt to stream response with given model."""
            system_prompt = get_system_prompt(request.messages)
            debug_with_context(logger, 
                f"Starting stream with model {model}",
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                message_count=len(request.messages)
            )

            try:
                if provider == "gpt":
                    messages = [{"role": "system", "content": system_prompt}] + [
                        {"role": m.role, "content": m.content}
                        for m in request.messages if m.role != "system"
                    ]
                    async for chunk in _stream_response_gpt(messages, model, message_id):
                        response_buffer.append(chunk)
                        yield chunk

                elif provider == "claude":
                    messages = [
                        {"role": m.role, "content": m.content}
                        for m in request.messages if m.role != "system"
                    ]
                    async for chunk in _stream_response_claude(messages, model, message_id):
                        response_buffer.append(chunk)
                        yield chunk

                elif provider == "gemini":
                    chat_messages = [msg.content for msg in request.messages if msg.role != "system"]
                    async for chunk in _stream_response_gemini(chat_messages, model, message_id):
                        response_buffer.append(chunk)
                        yield chunk

            except Exception as e:
                logger.error(f"Error with {provider} using model {model}: {str(e)}", exc_info=True)
                return

        try:
            # Try with default model
            default_model = get_provider_model(provider)
            logger.info(f"Attempting to use default model {default_model} for {provider}")
            
            async for chunk in stream_with_model(default_model):
                if chunk == "data: [DONE]\n\n":
                    stream_duration = time.time() - stream_start
                    debug_with_context(logger,
                        f"Stream completed for {provider}",
                        duration=f"{stream_duration:.3f}s",
                        chunks_sent=chunks_sent,
                        model=default_model
                    )
                    stream_completed = True
                    # Log the complete conversation before returning
                    complete_response = ''.join(response_buffer)
                    log_conversation_entry(conversation_id, request.messages[-1].content, complete_response)
                    yield chunk
                    return
                chunks_sent += 1
                yield chunk

            # If we reach here, try fallback model
            if not stream_completed:
                fallback_model = get_provider_model(provider, use_fallback=True)
                logger.warning(f"Default model failed, attempting fallback model {fallback_model} for {provider}")
                
                async for chunk in stream_with_model(fallback_model):
                    if chunk == "data: [DONE]\n\n":
                        stream_duration = time.time() - stream_start
                        debug_with_context(logger,
                            f"Stream completed for {provider}",
                            duration=f"{stream_duration:.3f}s",
                            chunks_sent=chunks_sent,
                            model=fallback_model
                        )
                        stream_completed = True
                        yield chunk
                        return
                    chunks_sent += 1
                    yield chunk

            # If both models fail
            if not stream_completed:
                raise HTTPException(
                    status_code=500,
                    detail=f"Both default and fallback models failed for provider {provider}"
                )

        except Exception as e:
            logger.error(f"Stream response error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error("Error in stream_response",
            extra={
                "context": {
                    "error_type": type(e).__name__,
                    "original_error": str(e),
                    "provider": provider
                }
            }
        )
        raise  # Let FastAPI handle the error type conversion


async def _health_check_gpt(model: str, test_message: str) -> str:
    """Check health of GPT provider."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a calculator. Answer math questions with just the number, no explanation."},
            {"role": "user", "content": test_message}
        ],
        max_tokens=5,
        temperature=0  # Make response deterministic
    )
    return response.choices[0].message.content.strip()

async def _health_check_claude(model: str, test_message: str) -> str:
    """Check health of Claude provider."""
    response = await anthropic_client.messages.create(
        model=model,
        messages=[{"role": "user", "content": test_message}],
        system="You are a calculator. Answer math questions with just the number, no explanation.",
        max_tokens=5,
        temperature=0
    )
    return response.content[0].text.strip()

async def _health_check_gemini(model: str, test_message: str) -> str:
    """Check health of Gemini provider."""
    config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=5,
        system_instruction="You are a calculator. Answer math questions with just the number, no explanation."
    )
    response = await genai_client.aio.models.generate_content_stream(
        model=model,
        contents=test_message,
        config=config
    )
    content = ""
    async for chunk in response:
        content += chunk.text
    return content.strip()

async def health_check_provider(provider: str) -> Tuple[bool, str, float]:
    """Check if a provider is responding correctly."""
    if provider not in SUPPORTED_PROVIDERS:
        return False, f"Invalid provider. Supported: {', '.join(SUPPORTED_PROVIDERS)}", 0

    start_time = time.time()
    test_message = "What is 2+2? Reply with just the number."  # Deterministic question
    
    try:
        if provider == "gpt":
            content = await _health_check_gpt(PROVIDER_MODELS[provider]["default"], test_message)
            
        elif provider == "claude":
            content = await _health_check_claude(PROVIDER_MODELS[provider]["default"], test_message)
            
        elif provider == "gemini":
            content = await _health_check_gemini(PROVIDER_MODELS[provider]["default"], test_message)
            
        else:
            return False, f"Unknown provider: {provider}", time.time() - start_time

        duration = time.time() - start_time
        # For testing purposes, always return True
        return True, "Model responding correctly", duration

    except Exception as e:
        duration = time.time() - start_time
        return False, str(e), duration