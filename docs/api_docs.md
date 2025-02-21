# API Documentation

**Base URL:** `http://localhost:3050` (default)

## Endpoints

### 1. Health Check

**GET `/health`**

Returns the overall system health status.

**Response:**
```json
{
    "status": "OK",
    "message": "System operational"
}
```

### 2. Provider Health Check

**GET `/health/{provider}`**

Check health status of a specific provider.

**Parameters:**
- `provider`: The AI provider to check (gpt, claude, gemini)

**Success Response:**
```json
{
    "status": "OK",
    "provider": "gpt",
    "message": "Using model: gpt-4o",
    "metrics": {
        "responseTime": 0.123
    }
}
```

**Error Response:**
```json
{
    "status": "ERROR",
    "provider": "gpt",
    "error": {
        "message": "Error message details"
    }
}
```

### 3. Chat Endpoint

**POST `/chat/{provider}`**

Stream chat responses from an AI provider.

**Headers:**
```
Accept: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

**Parameters:**
- `provider`: AI provider (gpt, claude, gemini)

**Request Body:**
```json
{
    "messages": [
        {
            "role": "user" | "assistant" | "system",
            "content": "message text",
            "timestamp": 1234567890,  // optional, unix timestamp
            "model": null  // optional, only used in responses
        }
    ]
}
```

Note: Each message in the array is a `ConversationMessage` object representing an entry in the conversation history.

- The `timestamp` field is optional in requests
- Model selection is handled automatically based on the provider:
  1. System first tries the provider's default model
  2. If default fails, system automatically tries the fallback model
  3. The actual model used is included in each response chunk
- The `model` field appears only in responses to track which model generated the content

**Response:**
Server-Sent Events (SSE) stream:
```
data: {
    "id": "gpt-1234567890",
    "delta": {
        "content": "partial response",
        "model": "gpt-4o"    // Indicates which model (default or fallback) is generating this response
    }
}
...
data: [DONE]
```

## Validation Rules

1. Messages:
   - Maximum length: 24000 characters per message
   - Maximum context: 50 messages total
   - Minimum length: 1 character
   - Required fields: role, content

2. Roles:
   - Allowed values: "user", "assistant", "system"
   - System messages are optional

## Error Responses

All errors follow this format:
```json
{
    "detail": "Error message"
}
```

Common error codes:
- **400**: Invalid request (wrong provider, invalid message format)
- **429**: Rate limit exceeded (500 requests per hour)
- **500**: Provider error or both default/fallback models failed
- **504**: Response timeout (30 seconds)

## Provider Configuration

Each provider uses these configurations:

**OpenAI (gpt)**
- Default model: gpt-4o
- Fallback model: gpt-4o-mini
- Temperature: 0.3
- Max tokens: 8192

**Anthropic (claude)**
- Default model: claude-3-5-sonnet-latest
- Fallback model: claude-3-5-haiku-latest
- Temperature: 0.3
- Max tokens: 8192

**Google (gemini)**
- Default model: gemini-2.0-flash
- Fallback model: gemini-1.5-pro
- Temperature: 0.3
- Max tokens: 8192

## Notes

- System messages are combined with provider-specific default prompts
- Model selection and fallback is handled automatically
- All responses are streamed using Server-Sent Events
- Response timeout is 30 seconds for all providers
- Rate limiting is applied across all providers