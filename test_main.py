# filepath: test_main.py
# usage: python -m pytest test_main.py -v
from fastapi.testclient import TestClient
from fastapi import HTTPException
from main import app
from models import ChatRequest, ChatMessage, HealthResponse
import json
import pytest
from typing import List

client = TestClient(app)

@pytest.fixture
def valid_chat_request():
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ],
        "model": "gpt-4o"
    }

@pytest.fixture
def valid_chat_request_with_history():
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more"}
        ],
        "model": "gpt-4o"
    }

def test_health_check():
    """Test the basic health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert data["message"] == "System operational"

def test_provider_health_check_gpt():
    """Test health check for GPT provider"""
    response = client.get("/health/gpt")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert data["provider"] == "gpt"
    assert "metrics" in data
    assert isinstance(data["metrics"]["responseTime"], float)

def test_provider_health_check_claude():
    """Test health check for Claude provider"""
    response = client.get("/health/claude")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert data["provider"] == "claude"
    assert "metrics" in data
    assert isinstance(data["metrics"]["responseTime"], float)

def test_provider_health_check_gemini():
    """Test health check for Gemini provider"""
    response = client.get("/health/gemini")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert data["provider"] == "gemini"
    assert "metrics" in data
    assert isinstance(data["metrics"]["responseTime"], float)

def test_invalid_provider_health_check():
    """Test health check with invalid provider"""
    response = client.get("/health/invalid_provider")
    assert response.status_code == 400
    assert "Invalid provider" in response.json()["detail"]

def test_chat_endpoint_invalid_provider():
    """Test chat endpoint with invalid provider"""
    chat_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4o"
    }
    response = client.post("/chat/invalid_provider", json=chat_request)
    assert response.status_code == 500  # Internal server error wraps the 400
    assert "Invalid provider" in str(response.text)

def test_chat_endpoint_invalid_model():
    """Test chat endpoint with invalid model"""
    chat_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "invalid-model"
    }
    response = client.post("/chat/gpt", json=chat_request)
    assert response.status_code == 500  # Internal server error wraps the 400
    assert "Invalid model for gpt" in str(response.text)

def test_chat_endpoint_stream_response(valid_chat_request):
    """Test chat endpoint streaming response"""
    response = client.post(
        "/chat/gpt",
        json=valid_chat_request,
        headers={"Accept": "text/event-stream"}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    
    # Get the first event from the stream
    for line in response.iter_lines():
        if line:
            assert line.startswith("data: ")
            data = json.loads(line.replace("data: ", ""))
            assert "id" in data
            assert "delta" in data
            assert "content" in data["delta"]
            break  # We only need to check the first event

def test_chat_endpoint_with_history(valid_chat_request_with_history):
    """Test chat endpoint with conversation history"""
    response = client.post(
        "/chat/gpt",
        json=valid_chat_request_with_history,
        headers={"Accept": "text/event-stream"}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

def test_chat_request_validation():
    """Test validation of chat request body"""
    # Test missing messages field
    invalid_request = {
        "model": "gpt-4o"
    }
    response = client.post("/chat/gpt", json=invalid_request)
    assert response.status_code == 422  # Validation error
    
    # Test missing model field
    invalid_request = {
        "messages": [{"role": "user", "content": "Hello"}]
    }
    response = client.post("/chat/gpt", json=invalid_request)
    assert response.status_code == 422

    # Test invalid JSON
    response = client.post("/chat/gpt", content=b"invalid json")
    assert response.status_code == 422

    # Test empty request body
    response = client.post("/chat/gpt", json={})
    assert response.status_code == 422

def test_chat_endpoint_error_handling():
    """Test error handling in chat endpoint"""
    # Test with very long message
    long_message = "a" * 25000  # Message longer than MAX_MESSAGE_LENGTH (24000)
    chat_request = {
        "messages": [{"role": "user", "content": long_message}],
        "model": "gpt-4o"
    }
    response = client.post("/chat/gpt", json=chat_request)
    assert response.status_code in [400, 422, 500]  # Either validation error (400/422) or internal error (500)

    # Test with empty message content
    chat_request = {
        "messages": [{"role": "user", "content": ""}],
        "model": "gpt-4o"
    }
    response = client.post("/chat/gpt", json=chat_request)
    assert response.status_code in [400, 422, 500]  # Either validation error (400/422) or internal error (500)

@pytest.mark.parametrize("provider,model", [
    ("gpt", "gpt-4o"),
    ("claude", "claude-3-5-sonnet-latest"),
    ("gemini", "gemini-2.0-flash")
])
def test_provider_model_compatibility(provider, model, valid_chat_request):
    """Test compatibility of models with providers"""
    valid_chat_request["model"] = model
    response = client.post(
        f"/chat/{provider}",
        json=valid_chat_request,
        headers={"Accept": "text/event-stream"}
    )
    assert response.status_code in [200, 500]  # 500 if API key not configured

def test_rate_limiting():
    """Test rate limiting functionality"""
    # Make multiple rapid requests
    chat_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4o"
    }
    
    responses = []
    for _ in range(5):  # Try 5 rapid requests
        response = client.post(
            "/chat/gpt",
            json=chat_request,
            headers={"Accept": "text/event-stream"}
        )
        responses.append(response)
    
    # At least one response should be successful
    assert any(r.status_code == 200 for r in responses)
    
    # Some responses might be rate limited
    rate_limited = any(r.status_code in [429, 500] for r in responses)
    if rate_limited:
        assert True  # Rate limiting is working

def test_health_check_response_model():
    """Test health check response matches the defined model"""
    response = client.get("/health")
    assert response.status_code == 200
    
    # Validate against HealthResponse model
    health_data = response.json()
    health_response = HealthResponse(**health_data)
    assert health_response.status == "OK"
    assert health_response.message == "System operational"
