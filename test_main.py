"""
Test suite for the chat API endpoints.
Tests cover health checks, chat functionality, and error handling across different providers.
"""

import json
import os
from typing import List, Callable
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv

from main import app

# Load environment variables
load_dotenv()

# Test client setup
client = TestClient(app)

# Constants
PROVIDER_MODEL_DEFAULT_MAP = {
    "gpt": "OPENAI_MODEL_DEFAULT",
    "claude": "ANTHROPIC_MODEL_DEFAULT",
    "gemini": "GEMINI_MODEL_DEFAULT"
}

PROVIDER_MODEL_FALLBACK_MAP = {
    "gpt": "OPENAI_MODEL_FALLBACK",
    "claude": "ANTHROPIC_MODEL_FALLBACK",
    "gemini": "GEMINI_MODEL_FALLBACK"
}

# Test configuration
def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on supported providers."""
    if "provider" in metafunc.fixturenames:
        providers_str = os.getenv("SUPPORTED_PROVIDERS", "gpt,claude,gemini")
        providers = [provider.strip() for provider in providers_str.split(",")]
        metafunc.parametrize("provider", providers)

# Fixtures
@pytest.fixture
def valid_chat_request():
    """Provides a basic valid chat request."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ],
        "model": os.getenv("OPENAI_MODEL_DEFAULT")
    }

@pytest.fixture
def valid_chat_request_with_history():
    """Provides a valid chat request with conversation history."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more"}
        ],
        "model": os.getenv("OPENAI_MODEL_DEFAULT")
    }

@pytest.fixture
def supported_providers() -> List[str]:
    """Returns list of supported providers from environment."""
    providers_str = os.getenv("SUPPORTED_PROVIDERS", "gpt,claude,gemini")
    return [provider.strip() for provider in providers_str.split(",")]

@pytest.fixture
def model_for_provider() -> Callable[[str, bool], str]:
    """Returns a function to get the appropriate model for a provider."""
    def _get_model(provider: str, use_fallback: bool = False) -> str:
        model_map = PROVIDER_MODEL_FALLBACK_MAP if use_fallback else PROVIDER_MODEL_DEFAULT_MAP
        if provider not in model_map:
            raise ValueError(f"Unknown provider: {provider}")
        return os.getenv(model_map[provider])
    return _get_model

@pytest.fixture
def env_vars():
    """Returns common environment variables used in tests."""
    return {
        "MAX_MESSAGE_LENGTH": int(os.getenv("MAX_MESSAGE_LENGTH", 24000)),
        "MAX_MESSAGES_IN_CONTEXT": int(os.getenv("MAX_MESSAGES_IN_CONTEXT", 50)),
        "RESPONSE_TIMEOUT": float(os.getenv("RESPONSE_TIMEOUT", 30.0)),
        "RATE_LIMIT_WINDOW_SECONDS": float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 3600.0))
    }

class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self):
        """Test the basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert data["message"] == "System operational"

    def test_provider_health_check(self, provider, supported_providers):
        """Test health check for each provider."""
        assert provider in supported_providers
        response = client.get(f"/health/{provider}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert data["provider"] == provider
        assert "metrics" in data
        assert isinstance(data["metrics"]["responseTime"], float)

    def test_invalid_provider_health_check(self):
        """Test health check with invalid provider."""
        response = client.get("/health/invalid_provider")
        assert response.status_code == 400
        assert "Invalid provider" in response.json()["detail"]

class TestChatEndpoints:
    """Tests for chat functionality endpoints."""

    def test_chat_endpoint_stream_response(self, provider, valid_chat_request, 
                                         model_for_provider, supported_providers):
        """Test chat endpoint streaming response."""
        assert provider in supported_providers
        valid_chat_request["model"] = model_for_provider(provider, use_fallback=False)
        
        response = client.post(
            f"/chat/{provider}",
            json=valid_chat_request,
            headers={"Accept": "text/event-stream"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        
        self._verify_stream_response(response)

    def test_chat_endpoint_with_history(self, provider, valid_chat_request_with_history,
                                      model_for_provider, supported_providers):
        """Test chat endpoint with conversation history."""
        assert provider in supported_providers
        valid_chat_request_with_history["model"] = model_for_provider(provider, use_fallback=False)
        
        response = client.post(
            f"/chat/{provider}",
            json=valid_chat_request_with_history,
            headers={"Accept": "text/event-stream"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    @staticmethod
    def _verify_stream_response(response):
        """Helper method to verify streaming response format."""
        for line in response.iter_lines():
            if line:
                assert line.startswith("data: ")
                data = json.loads(line.replace("data: ", ""))
                assert "id" in data
                assert "delta" in data
                assert "content" in data["delta"]
                break

class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_chat_endpoint_invalid_provider(self, valid_chat_request):
        """Test chat endpoint with invalid provider."""
        response = client.post("/chat/invalid_provider", json=valid_chat_request)
        assert response.status_code == 500
        assert "Invalid provider" in str(response.text)

    def test_chat_endpoint_invalid_model(self, provider, supported_providers):
        """Test chat endpoint with invalid model."""
        assert provider in supported_providers
        chat_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "invalid-model"
        }
        response = client.post(f"/chat/{provider}", json=chat_request)
        assert response.status_code == 500
        assert f"Invalid model for {provider}" in str(response.text)

    def test_chat_request_validation(self, provider, model_for_provider, supported_providers):
        """Test validation of chat request body."""
        assert provider in supported_providers
        self._test_missing_fields(provider, model_for_provider)
        self._test_invalid_format(provider)

    def _test_missing_fields(self, provider, model_for_provider):
        """Helper method to test missing required fields."""
        invalid_request = {"model": model_for_provider(provider)}
        response = client.post(f"/chat/{provider}", json=invalid_request)
        assert response.status_code == 422

        invalid_request = {"messages": [{"role": "user", "content": "Hello"}]}
        response = client.post(f"/chat/{provider}", json=invalid_request)
        assert response.status_code == 422

    def _test_invalid_format(self, provider):
        """Helper method to test invalid request formats."""
        response = client.post(f"/chat/{provider}", content=b"invalid json")
        assert response.status_code == 422

        response = client.post(f"/chat/{provider}", json={})
        assert response.status_code == 422

class TestEnvironmentValidation:
    """Tests for environment variable validation."""
    
    def test_required_env_vars_present(self):
        """Test that required environment variables are present."""
        required_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "MAX_MESSAGE_LENGTH",
            "MAX_MESSAGES_IN_CONTEXT",
            "RESPONSE_TIMEOUT",
            "RATE_LIMIT_WINDOW_SECONDS"
        ]
        
        for var in required_vars:
            assert os.getenv(var) is not None, f"Missing required environment variable: {var}"
            
    def test_numeric_env_vars_valid(self):
        """Test that numeric environment variables can be properly parsed."""
        numeric_vars = {
            "MAX_MESSAGE_LENGTH": int,
            "MAX_MESSAGES_IN_CONTEXT": int,
            "RESPONSE_TIMEOUT": float,
            "RATE_LIMIT_WINDOW_SECONDS": float,
            "RATE_LIMIT_MAX_REQUESTS": int
        }
        
        for var, type_func in numeric_vars.items():
            value = os.getenv(var)
            assert value is not None, f"Missing numeric environment variable: {var}"
            try:
                type_func(value)
            except ValueError:
                pytest.fail(f"Environment variable {var} must be a valid {type_func.__name__}")

class TestMessageLimits:
    """Tests for message size and count limits."""
    
    def test_content_length_limit(self, provider, env_vars, model_for_provider):
        """Test that messages exceeding MAX_MESSAGE_LENGTH are rejected."""
        max_length = env_vars["MAX_MESSAGE_LENGTH"]
        oversized_content = "x" * (max_length + 1)
        
        chat_request = {
            "messages": [
                {"role": "user", "content": oversized_content}
            ],
            "model": model_for_provider(provider)
        }
        
        response = client.post(
            f"/chat/{provider}",
            json=chat_request
        )
        
        assert response.status_code == 422
        error_detail = response.json()["detail"][0]
        assert error_detail["msg"] == f"Value error, Message exceeds maximum length of {max_length} characters"
        assert error_detail["loc"] == ["body", "messages", 0, "content"]
        
    def test_message_count_limit(self, provider, env_vars, model_for_provider):
        """Test that requests exceeding MAX_MESSAGES_IN_CONTEXT are rejected."""
        max_messages = env_vars["MAX_MESSAGES_IN_CONTEXT"]
        
        chat_request = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ] * (max_messages + 1),
            "model": model_for_provider(provider)
        }
        
        response = client.post(
            f"/chat/{provider}",
            json=chat_request
        )
        
        assert response.status_code == 422
        error_detail = response.json()["detail"][0]
        assert error_detail["msg"] == f"Value error, Conversation exceeds maximum of {max_messages} messages"
        assert error_detail["loc"] == ["body", "messages"]

class TestStreamingResponse:
    """Tests for streaming response behavior."""
    
    def test_streaming_response_timeout(self, provider, valid_chat_request, 
                                      model_for_provider, env_vars):
        """Test that streaming responses complete within the configured timeout."""
        valid_chat_request["model"] = model_for_provider(provider)
        timeout = env_vars["RESPONSE_TIMEOUT"]
        
        start_time = time.time()
        response = client.post(
            f"/chat/{provider}",
            json=valid_chat_request,
            headers={"Accept": "text/event-stream"}
        )
        
        # Collect all streaming responses
        full_response = ""
        for line in response.iter_lines():
            if line:
                # Handle both string and bytes types
                line_text = line if isinstance(line, str) else line.decode('utf-8')
                full_response += line_text
        
        completion_time = time.time() - start_time
        
        assert response.status_code == 200
        assert completion_time <= timeout, (
            f"Response took {completion_time:.2f}s, exceeding {timeout}s timeout"
        )
        assert full_response, "Response should not be empty"

# Additional test classes can be added here for other functionality groups

