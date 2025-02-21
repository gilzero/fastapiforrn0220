# `test_aiproviders.py` Documentation

This file contains unit and integration tests for the `aiproviders.py` module, which handles interactions with various AI providers (OpenAI GPT, Anthropic Claude, and Google Gemini).

## Test Data

-   **`TEST_MESSAGE`**: A sample `ConversationMessage` with the role `USER` and content "Test message".
-   **`TEST_SYSTEM_MESSAGE`**: A sample `ConversationMessage` with the role `SYSTEM` and content "Test system prompt".
-   **`TEST_REQUEST`**: A sample `ChatRequest` containing both `TEST_SYSTEM_MESSAGE` and `TEST_MESSAGE`.

## Helper Function Tests

These tests verify the behavior of utility functions within `aiproviders.py`.

-   **`test_get_system_prompt_with_system_message()`**:  Asserts that `get_system_prompt` correctly returns the content of a provided system message.
-   **`test_get_system_prompt_without_system_message()`**: Asserts that `get_system_prompt` returns a generic system prompt when no system message is present in the input list.
-   **`test_get_system_prompt_empty_messages()`**: Asserts that `get_system_prompt` returns the `GENERIC_SYSTEM_PROMPT` when given an empty list of messages.
-   **`test_get_system_prompt_empty_content()`**: Asserts that `get_system_prompt` returns the `GENERIC_SYSTEM_PROMPT` when the system message has only minimal content (e.g. only ".").
-   **`test_get_system_prompt_whitespace_content()`**: Asserts that get_system_prompt` returns the `GENERIC_SYSTEM_PROMPT` when the system message content consist of only whitespaces and a minimal valid character (e.g. " . \n  \t  . ").
-   **`test_get_provider_model_default()`**: Asserts that `get_provider_model` returns the correct default model for a given provider ("gpt").
-   **`test_get_provider_model_fallback()`**: Asserts that `get_provider_model` returns the correct fallback model for a given provider ("gpt") when `use_fallback=True`.
-   **`test_get_provider_model_invalid()`**: Asserts that `get_provider_model` raises an exception when an invalid provider name is given.
-  **`test_conversation_message_validation()`**: Validates the `ConversationMessage` model's input validation, ensuring that an exception is raised if the message content is empty or consists only of whitespace.

## Stream Response Tests

These tests use `pytest.mark.asyncio` to verify the asynchronous `stream_response` function. They utilize `unittest.mock`'s `AsyncMock` and `MagicMock` to simulate API responses.

-   **`test_stream_response_gpt()`**: Tests streaming from the GPT provider.  Mocks the OpenAI client and asserts that the stream returns chunks containing the expected "Test response" and ends with "[DONE]".
-   **`test_stream_response_claude()`**: Tests streaming from the Claude provider. Mocks the Anthropic client and asserts that the stream contains "Test" and ends with "[DONE]".
-   **`test_stream_response_gemini()`**: Tests streaming from the Gemini provider. Mocks the Google GenAI client and asserts that the stream returns chunks containing "Test response" and ends with "[DONE]".
-   **`test_stream_response_invalid_provider()`**: Asserts that `stream_response` raises an exception when an invalid provider is requested.
-   **`test_stream_response_empty_stream()`**: Tests the handling of an empty stream from the AI provider, ensuring that "[DONE]" is still returned.
-   **`test_stream_response_malformed_chunk()`**: Tests the handling of malformed chunks (missing expected attributes) from the AI provider. It verifies that the code doesn't crash and still returns "[DONE]". This is a crucial test for robustness.

## Health Check Tests

These tests use `pytest.mark.asyncio` to verify the asynchronous `check_provider_health` function.

-   **`test_check_provider_health_success()`**: Tests a successful health check.  Mocks the OpenAI client to return a valid "OK" response and asserts that the function returns `True`, a success message, and a duration.
-   **`test_check_provider_health_failure()`**: Tests a failed health check.  Mocks the OpenAI client to raise an exception and asserts that the function returns `False`, an error message, and a duration.