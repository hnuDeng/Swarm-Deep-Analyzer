"""
Mock OpenAI client for testing the Swarm framework.

Provides MockOpenAIClient that mimics the openai.OpenAI() interface
with configurable responses for chat.completions.create().
"""
import json
from swarm.types import ChatCompletionMessage, ChatCompletionMessageToolCall, Function
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChunkChoice,
    ChoiceDelta,
)


def create_mock_response(message, function_calls=[], model="gpt-4o"):
    """Create a mock ChatCompletion response."""
    role = message.get("role", "assistant")
    content = message.get("content", "")
    tool_calls = (
        [
            ChatCompletionMessageToolCall(
                id=f"mock_tc_id_{i}",
                type="function",
                function=Function(
                    name=call.get("name", ""),
                    arguments=json.dumps(call.get("args", {})),
                ),
            )
            for i, call in enumerate(function_calls)
        ]
        if function_calls
        else None
    )

    return ChatCompletion(
        id="mock_cc_id",
        created=1234567890,
        model=model,
        object="chat.completion",
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role=role, content=content, tool_calls=tool_calls
                ),
                finish_reason="stop",
                index=0,
            )
        ],
    )


def create_mock_stream_chunks(content_parts):
    """Create a list of mock ChatCompletionChunk objects for streaming tests."""
    chunks = []
    # First chunk with role
    chunks.append(ChatCompletionChunk(
        id="mock_chunk_id",
        created=1234567890,
        model="gpt-4o",
        object="chat.completion.chunk",
        choices=[ChunkChoice(
            index=0,
            delta=ChoiceDelta(role="assistant", content=None),
            finish_reason=None,
        )],
    ))
    # Content chunks
    for part in content_parts:
        chunks.append(ChatCompletionChunk(
            id="mock_chunk_id",
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk",
            choices=[ChunkChoice(
                index=0,
                delta=ChoiceDelta(content=part, role=None),
                finish_reason=None,
            )],
        ))
    # Final chunk with finish reason
    chunks.append(ChatCompletionChunk(
        id="mock_chunk_id",
        created=1234567890,
        model="gpt-4o",
        object="chat.completion.chunk",
        choices=[ChunkChoice(
            index=0,
            delta=ChoiceDelta(content=None, role=None),
            finish_reason="stop",
        )],
    ))
    return chunks


class MockCompletions:
    """Mock for client.chat.completions."""

    def __init__(self):
        self._responses = []
        self._stream_responses = []
        self._call_count = 0

    def set_response(self, response: ChatCompletion):
        self._responses = [response]
        self._call_count = 0

    def set_sequential_responses(self, responses: list):
        self._responses = responses
        self._call_count = 0

    def set_stream_responses(self, stream_responses: list):
        """Set responses for streaming mode. Each entry is a list of chunks."""
        self._stream_responses = stream_responses
        self._call_count = 0

    def create(self, **kwargs):
        stream = kwargs.get("stream", False)
        if stream:
            if self._call_count < len(self._stream_responses):
                chunks = self._stream_responses[self._call_count]
                self._call_count += 1
                return iter(chunks)
            return iter([])
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return self._responses[-1] if self._responses else None


class MockChat:
    """Mock for client.chat."""

    def __init__(self):
        self.completions = MockCompletions()


class MockOpenAIClient:
    """
    Mock OpenAI client that mimics the openai.OpenAI() interface.
    Supports client.chat.completions.create() calls.
    """

    def __init__(self):
        self.chat = MockChat()

    def set_response(self, response: ChatCompletion):
        self.chat.completions.set_response(response)

    def set_sequential_responses(self, responses: list):
        self.chat.completions.set_sequential_responses(responses)

    def set_stream_responses(self, stream_responses: list):
        self.chat.completions.set_stream_responses(stream_responses)


def _make_mock_client():
    """Create a fresh MockOpenAIClient with a default response."""
    client = MockOpenAIClient()
    client.set_response(
        create_mock_response({"role": "assistant", "content": "default response"})
    )
    return client
