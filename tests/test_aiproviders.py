# filepath: tests/test_aiproviders.py
import pytest
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch
from models import ChatRequest, ConversationMessage, MessageRole
from aiproviders import (
    get_system_prompt,
    get_provider_model,
    stream_response,
    check_provider_health,
    PROVIDER_MODELS,
    SUPPORTED_PROVIDERS,
    GENERIC_SYSTEM_PROMPT
)

# Test data
TEST_MESSAGE = ConversationMessage(
    role=MessageRole.USER,
    content="Test message",
    timestamp=int(datetime.now(UTC).timestamp())
)

TEST_SYSTEM_MESSAGE = ConversationMessage(
    role=MessageRole.SYSTEM,
    content="Test system prompt",
    timestamp=int(datetime.now(UTC).timestamp())
)

TEST_REQUEST = ChatRequest(messages=[TEST_SYSTEM_MESSAGE, TEST_MESSAGE])

# Helper function tests
def test_get_system_prompt_with_system_message():
    """Test get_system_prompt when system message is provided"""
    prompt = get_system_prompt([TEST_SYSTEM_MESSAGE, TEST_MESSAGE])
    assert prompt == TEST_SYSTEM_MESSAGE.content

def test_get_system_prompt_without_system_message():
    """Test get_system_prompt falls back to default when no system message"""
    prompt = get_system_prompt([TEST_MESSAGE])
    assert "helpful ai assistant" in prompt.lower()

def test_get_provider_model_default():
    """Test get_provider_model returns correct default model"""
    model = get_provider_model("gpt")
    assert model == PROVIDER_MODELS["gpt"]["default"]

def test_get_provider_model_fallback():
    """Test get_provider_model returns correct fallback model"""
    model = get_provider_model("gpt", use_fallback=True)
    assert model == PROVIDER_MODELS["gpt"]["fallback"]

def test_get_provider_model_invalid():
    """Test get_provider_model raises error for invalid provider"""
    with pytest.raises(Exception):
        get_provider_model("invalid_provider")

def test_get_system_prompt_empty_messages():
    """Test get_system_prompt with empty messages list"""
    prompt = get_system_prompt([])
    assert prompt == GENERIC_SYSTEM_PROMPT

def test_get_system_prompt_empty_content():
    """Test get_system_prompt with empty system message content"""
    minimal_system_msg = ConversationMessage(
        role=MessageRole.SYSTEM,
        content=" . ",  # Minimal valid content
        timestamp=int(datetime.now(UTC).timestamp())
    )
    prompt = get_system_prompt([minimal_system_msg])
    assert prompt == GENERIC_SYSTEM_PROMPT

def test_get_system_prompt_whitespace_content():
    """Test get_system_prompt with whitespace-only system message"""
    minimal_msg = ConversationMessage(
        role=MessageRole.SYSTEM,
        content=" . \n  \t  . ",  # Minimal valid content with whitespace
        timestamp=int(datetime.now(UTC).timestamp())
    )
    prompt = get_system_prompt([minimal_msg])
    assert prompt == GENERIC_SYSTEM_PROMPT

def test_conversation_message_validation():
    """Test that ConversationMessage properly validates content"""
    with pytest.raises(Exception) as exc_info:
        ConversationMessage(
            role=MessageRole.SYSTEM,
            content="",
            timestamp=int(datetime.now(UTC).timestamp())
        )
    assert "Message content cannot be empty" in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        ConversationMessage(
            role=MessageRole.SYSTEM,
            content="   \n  \t  ",
            timestamp=int(datetime.now(UTC).timestamp())
        )
    assert "Message content cannot be empty" in str(exc_info.value)

# Stream response tests
@pytest.mark.asyncio
async def test_stream_response_gpt():
    """Test GPT stream response"""
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Test response"

    mock_stream = MagicMock()
    mock_stream.__iter__ = lambda _: iter([mock_chunk])

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_stream

    with patch('aiproviders.client', mock_client):
        responses = []
        async for chunk in stream_response(TEST_REQUEST, "gpt"):
            responses.append(chunk)

        assert len(responses) > 0
        assert "Test response" in responses[0]
        assert "[DONE]" in responses[-1]

@pytest.mark.asyncio
async def test_stream_response_claude():
    """Test Claude stream response"""
    mock_stream = MagicMock()
    mock_stream.text_stream = ["Test", "response"]

    class AsyncContextManager:
        async def __aenter__(self):
            return mock_stream
        async def __aexit__(self, *args):
            pass

    mock_client = AsyncMock()
    mock_client.messages.stream.return_value = AsyncContextManager()

    with patch('aiproviders.anthropic_client', mock_client):
        responses = []
        async for chunk in stream_response(TEST_REQUEST, "claude"):
            responses.append(chunk)

        assert len(responses) > 0
        assert any("Test" in r for r in responses)
        assert "[DONE]" in responses[-1]

@pytest.mark.asyncio
async def test_stream_response_gemini():
    """Test Gemini stream response"""
    mock_chunk = MagicMock()
    mock_chunk.text = "Test response"

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = [mock_chunk]

    with patch('aiproviders.genai_client', mock_client):
        responses = []
        async for chunk in stream_response(TEST_REQUEST, "gemini"):
            responses.append(chunk)

        assert len(responses) > 0
        assert "Test response" in responses[0]
        assert "[DONE]" in responses[-1]

@pytest.mark.asyncio
async def test_stream_response_invalid_provider():
    """Test stream response with invalid provider"""
    with pytest.raises(Exception):
        async for _ in stream_response(TEST_REQUEST, "invalid_provider"):
            pass

@pytest.mark.asyncio
async def test_stream_response_empty_stream():
    """Test handling of empty stream response"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = MagicMock(__iter__=lambda _: iter([]))

    with patch('aiproviders.client', mock_client):
        responses = []
        async for chunk in stream_response(TEST_REQUEST, "gpt"):
            responses.append(chunk)
        
        assert len(responses) > 0
        assert "[DONE]" in responses[-1]

@pytest.mark.asyncio
async def test_stream_response_malformed_chunk():
    """Test handling of malformed chunks in stream"""
    mock_chunk = MagicMock()
    # Simulate malformed chunk with missing attributes
    del mock_chunk.choices
    
    mock_stream = MagicMock()
    mock_stream.__iter__ = lambda _: iter([mock_chunk])

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_stream

    with patch('aiproviders.client', mock_client):
        responses = []
        async for chunk in stream_response(TEST_REQUEST, "gpt"):
            responses.append(chunk)
        
        assert len(responses) > 0
        assert "[DONE]" in responses[-1]

# Health check tests
@pytest.mark.asyncio
async def test_check_provider_health_success():
    """Test successful health check"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "OK"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch('aiproviders.client', mock_client):
        success, message, duration = await check_provider_health("gpt")
        assert success is True
        assert "responding correctly" in message
        assert isinstance(duration, float)

@pytest.mark.asyncio
async def test_check_provider_health_failure():
    """Test failed health check"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = Exception("Test error")

    with patch('aiproviders.client', mock_client):
        success, message, duration = await check_provider_health("gpt")
        assert success is False
        assert "Test error" in message
        assert isinstance(duration, float) 