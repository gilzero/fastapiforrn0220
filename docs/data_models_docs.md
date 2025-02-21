## data models docs

## Data Models Documentation

### ConversationMessage

Represents an individual message entry in a conversation.

- **role**: The role of the message sender. Can be one of `user`, `assistant`, or `system`.
- **content**: The content of the message. Must be a non-empty string and cannot exceed the maximum length defined by `MAX_MESSAGE_LENGTH`.
- **timestamp**: (Optional) The timestamp of the message.
- **model**: (Optional) The model used for generating the message. This field is populated in responses for tracking purposes.

Note: System messages are optional and will be combined with provider-specific system prompts. If no system message is provided, default system prompts will be used for each provider.

### ChatRequest

Represents a request to the chat endpoint.

- **messages**: A list of `ConversationMessage` objects representing the conversation history. The number of messages cannot exceed the maximum defined by `MAX_MESSAGES_IN_CONTEXT`.

Note: The model selection is handled automatically based on the provider specified in the endpoint URL (`/chat/{provider}`). The system will attempt to use the default model for the provider and fall back to an alternative model if necessary.

### Chat Response Structure

While we use formal Pydantic models for requests and health checks, the chat response is intentionally implemented as a direct JSON structure for several reasons:

1. **Streaming Nature**: The response is streamed as Server-Sent Events (SSE), making a formal model less beneficial
2. **Provider Flexibility**: Different AI providers may have varying response formats or additional metadata
3. **Performance**: For high-frequency streaming responses, direct JSON serialization is more efficient
4. **Future Compatibility**: A flexible structure allows easier integration of new providers or response formats

The response follows this consistent structure:
```json
{
    "id": "provider-timestamp",
    "delta": {
        "content": "partial response text",
        "model": "model identifier"
    }
}
```

Followed by a completion marker:
```
data: [DONE]
```

### HealthResponse

Represents the response from a health check endpoint.

- **status**: The status of the health check. Can be `OK` or `ERROR`.
- **message**: (Optional) A message providing additional information, including the current default model being used.
- **provider**: (Optional) The provider being checked.
- **metrics**: (Optional) A dictionary containing metrics such as response time.
- **error**: (Optional) A dictionary containing error details.

### Provider Selection

The provider is specified in the endpoint URL (e.g., `/chat/gpt` or `/chat/claude`). Available providers are configured via the `SUPPORTED_PROVIDERS` environment variable. The system will automatically select the appropriate model based on the provider:

- Each provider has a default model (e.g., `gpt-4o` for GPT)
- If the default model fails, the system automatically tries a fallback model (e.g., `gpt-4o-mini`)
- The actual model used will be included in the response message's `model` field