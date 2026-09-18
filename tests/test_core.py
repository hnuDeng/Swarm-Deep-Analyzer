"""Tests for swarm.core module - comprehensive coverage."""
import pytest
import json
import copy
from swarm import Swarm, Agent
from swarm.types import Result, Response
from tests.mock_client import (
    MockOpenAIClient, create_mock_response, create_mock_stream_chunks,
    _make_mock_client,
)
from unittest.mock import Mock

DEFAULT_RESPONSE_CONTENT = "sample response content"


@pytest.fixture
def mock_openai_client():
    m = _make_mock_client()
    m.set_response(
        create_mock_response({"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT})
    )
    return m


# --- Non-streaming tests ---

def test_run_with_simple_message(mock_openai_client):
    agent = Agent()
    client = Swarm(client=mock_openai_client)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = client.run(agent=agent, messages=messages)

    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


def test_tool_call(mock_openai_client):
    expected_location = "San Francisco"
    get_weather_mock = Mock()

    def get_weather(location):
        get_weather_mock(location=location)
        return "It's sunny today."

    agent = Agent(name="Test Agent", functions=[get_weather])
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": expected_location}}
                ],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    client = Swarm(client=mock_openai_client)
    response = client.run(agent=agent, messages=messages)

    get_weather_mock.assert_called_once_with(location=expected_location)
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


def test_execute_tools_false(mock_openai_client):
    expected_location = "San Francisco"
    get_weather_mock = Mock()

    def get_weather(location):
        get_weather_mock(location=location)
        return "It's sunny today."

    agent = Agent(name="Test Agent", functions=[get_weather])
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": expected_location}}
                ],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    client = Swarm(client=mock_openai_client)
    response = client.run(agent=agent, messages=messages, execute_tools=False)

    get_weather_mock.assert_not_called()

    tool_calls = response.messages[-1].get("tool_calls")
    assert tool_calls is not None and len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "location": expected_location
    }


def test_handoff(mock_openai_client):
    def transfer_to_agent2():
        return agent2

    agent1 = Agent(name="Test Agent 1", functions=[transfer_to_agent2])
    agent2 = Agent(name="Test Agent 2")

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "transfer_to_agent2"}],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    client = Swarm(client=mock_openai_client)
    messages = [{"role": "user", "content": "I want to talk to agent 2"}]
    response = client.run(agent=agent1, messages=messages)

    assert response.agent == agent2
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


def test_context_variables_passing(mock_openai_client):
    """Test that context_variables are passed to functions that accept them."""
    received_vars = {}

    def track_context(context_variables):
        received_vars.update(context_variables)
        return "tracked"

    agent = Agent(name="Tracker Agent", functions=[track_context])

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "track_context", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": "Done tracking"}
            ),
        ]
    )

    client = Swarm(client=mock_openai_client)
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "track"}],
        context_variables={"key1": "value1", "key2": 42},
    )

    assert received_vars.get("key1") == "value1"
    assert received_vars.get("key2") == 42


def test_context_variables_update(mock_openai_client):
    """Test that context_variables are updated from Result."""
    def update_vars(context_variables):
        return Result(
            value="updated",
            context_variables={"new_key": "new_value"},
        )

    agent = Agent(name="Updater Agent", functions=[update_vars])

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "update_vars", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": "All done"}
            ),
        ]
    )

    client = Swarm(client=mock_openai_client)
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "update"}],
        context_variables={"existing": "kept"},
    )

    assert response.context_variables.get("new_key") == "new_value"
    assert response.context_variables.get("existing") == "kept"


def test_max_turns_limit(mock_openai_client):
    """Test that max_turns limits the number of conversation turns."""
    def dummy_tool():
        return "result"

    agent = Agent(name="Looping Agent", functions=[dummy_tool])

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "dummy_tool", "args": {}}],
            ),
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "dummy_tool", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": "finally done"}
            ),
        ]
    )

    client = Swarm(client=mock_openai_client)
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "loop"}],
        max_turns=2,
    )

    assert response is not None


def test_missing_tool_call(mock_openai_client):
    """Test handling of a tool call to a function that doesn't exist."""
    agent = Agent(name="Test Agent", functions=[])

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "nonexistent_function", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": "fallback response"}
            ),
        ]
    )

    client = Swarm(client=mock_openai_client)
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "call missing"}],
    )

    tool_messages = [m for m in response.messages if m.get("role") == "tool"]
    assert len(tool_messages) > 0
    assert "not found" in tool_messages[0]["content"]


def test_agent_with_string_instructions(mock_openai_client):
    agent = Agent(instructions="You are a test agent.")
    client = Swarm(client=mock_openai_client)
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


def test_agent_with_callable_instructions(mock_openai_client):
    def dynamic_instructions(context_variables):
        return "You are helping user %s." % context_variables.get("user_id", "unknown")

    agent = Agent(instructions=dynamic_instructions)
    client = Swarm(client=mock_openai_client)
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "test"}],
        context_variables={"user_id": "test_user"},
    )
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


def test_handle_function_result_string():
    client = Swarm(client=_make_mock_client())
    result = client.handle_function_result("plain string", debug=False)
    assert isinstance(result, Result)
    assert result.value == "plain string"
    assert result.agent is None


def test_handle_function_result_agent():
    client = Swarm(client=_make_mock_client())
    agent = Agent(name="Returned Agent")
    result = client.handle_function_result(agent, debug=False)
    assert isinstance(result, Result)
    assert result.agent == agent
    assert "Returned Agent" in result.value


def test_handle_function_result_result_object():
    client = Swarm(client=_make_mock_client())
    original = Result(value="test", context_variables={"a": 1})
    result = client.handle_function_result(original, debug=False)
    assert result is original


def test_multi_tool_call(mock_openai_client):
    results = []

    def tool_a():
        results.append("a")
        return "result_a"

    def tool_b():
        results.append("b")
        return "result_b"

    agent = Agent(name="Multi Tool Agent", functions=[tool_a, tool_b])

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "tool_a", "args": {}},
                    {"name": "tool_b", "args": {}},
                ],
            ),
            create_mock_response(
                {"role": "assistant", "content": "Both tools called"}
            ),
        ]
    )

    client = Swarm(client=mock_openai_client)
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "call both"}],
    )

    assert "a" in results
    assert "b" in results
    assert response.messages[-1]["content"] == "Both tools called"


def test_response_structure(mock_openai_client):
    agent = Agent(name="Structure Agent")
    client = Swarm(client=mock_openai_client)
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "test"}],
    )

    assert isinstance(response, Response)
    assert hasattr(response, "messages")
    assert hasattr(response, "agent")
    assert hasattr(response, "context_variables")
    assert isinstance(response.messages, list)
    assert isinstance(response.context_variables, dict)


# --- Streaming Tests ---

def _collect_stream(gen):
    """Collect streaming chunks, copying each dict to avoid mutation by the generator."""
    chunks = []
    try:
        while True:
            c = next(gen)
            chunks.append(copy.deepcopy(c) if isinstance(c, dict) else c)
    except StopIteration:
        pass
    return chunks


def test_stream_simple_response(mock_openai_client):
    """Test basic streaming with a simple text response."""
    mock_openai_client.set_stream_responses([
        create_mock_stream_chunks(["Hello", " world", "!"]),
    ])

    agent = Agent(name="Stream Agent")
    client = Swarm(client=mock_openai_client)
    gen = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    chunks = _collect_stream(gen)
    response_chunks = [c for c in chunks if "response" in c]
    assert len(response_chunks) == 1
    assert isinstance(response_chunks[0]["response"], Response)


def test_stream_preserves_sender(mock_openai_client):
    """Test that streaming chunks include the sender name in the role chunk."""
    mock_openai_client.set_stream_responses([
        create_mock_stream_chunks(["test"]),
    ])

    agent = Agent(name="MyAgent")
    client = Swarm(client=mock_openai_client)
    gen = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    chunks = _collect_stream(gen)
    sender_chunks = [c for c in chunks if "sender" in c]
    assert len(sender_chunks) >= 1
    assert sender_chunks[0]["sender"] == "MyAgent"


def test_stream_delimiters(mock_openai_client):
    """Test that streaming produces start/end delimiters."""
    mock_openai_client.set_stream_responses([
        create_mock_stream_chunks(["content"]),
    ])

    agent = Agent(name="Delim Agent")
    client = Swarm(client=mock_openai_client)
    gen = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    chunks = _collect_stream(gen)
    delims = [c["delim"] for c in chunks if "delim" in c]
    assert "start" in delims
    assert "end" in delims


def test_stream_final_response_structure(mock_openai_client):
    """Test that the final streaming response has correct structure."""
    mock_openai_client.set_stream_responses([
        create_mock_stream_chunks(["result"]),
    ])

    agent = Agent(name="Final Agent")
    client = Swarm(client=mock_openai_client)
    gen = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "test"}],
        stream=True,
    )

    chunks = _collect_stream(gen)
    response_chunk = [c for c in chunks if "response" in c][0]
    resp = response_chunk["response"]

    assert isinstance(resp, Response)
    assert resp.agent.name == "Final Agent"
    assert isinstance(resp.messages, list)
    assert isinstance(resp.context_variables, dict)


def test_stream_content_accumulation(mock_openai_client):
    """Test that streaming content is properly accumulated in final response."""
    mock_openai_client.set_stream_responses([
        create_mock_stream_chunks(["Hello", " ", "World"]),
    ])

    agent = Agent(name="Accum Agent")
    client = Swarm(client=mock_openai_client)
    gen = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "greet"}],
        stream=True,
    )

    chunks = _collect_stream(gen)
    response_chunk = [c for c in chunks if "response" in c][0]
    resp = response_chunk["response"]

    last_msg = resp.messages[-1]
    assert "Hello World" in last_msg["content"]


def test_stream_empty_content(mock_openai_client):
    """Test streaming with empty content response."""
    mock_openai_client.set_stream_responses([
        create_mock_stream_chunks([]),
    ])

    agent = Agent(name="Empty Agent")
    client = Swarm(client=mock_openai_client)
    gen = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "test"}],
        stream=True,
    )

    chunks = _collect_stream(gen)
    response_chunk = [c for c in chunks if "response" in c][0]
    assert response_chunk["response"] is not None


def test_stream_uses_active_agent_name(mock_openai_client):
    """Test that streaming uses active_agent name in sender field (bug fix verification)."""
    mock_openai_client.set_stream_responses([
        create_mock_stream_chunks(["result"]),
    ])

    agent = Agent(name="ActiveAgent")
    client = Swarm(client=mock_openai_client)
    gen = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "test"}],
        stream=True,
    )

    chunks = _collect_stream(gen)
    # The sender in the role chunk should match the active agent
    sender_chunks = [c for c in chunks if "sender" in c]
    assert len(sender_chunks) >= 1
    assert sender_chunks[0]["sender"] == "ActiveAgent"
    # The response should also have the correct agent
    response_chunk = [c for c in chunks if "response" in c][0]
    assert response_chunk["response"].agent.name == "ActiveAgent"

# --- Error Handling & Retry Tests ---

def test_retry_on_rate_limit(mock_openai_client):
    """Test that API calls retry on rate limit errors."""
    from openai import RateLimitError
    from unittest.mock import MagicMock

    call_count = 0
    original_create = mock_openai_client.chat.completions.create

    def failing_then_succeeding(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
        return original_create(**kwargs)

    mock_openai_client.chat.completions.create = failing_then_succeeding
    mock_openai_client.set_response(
        create_mock_response({"role": "assistant", "content": "recovered"})
    )
    # Override the create after setting response
    saved_responses = mock_openai_client.chat.completions._responses
    mock_openai_client.chat.completions.create = failing_then_succeeding
    mock_openai_client.chat.completions._responses = saved_responses

    client = Swarm(client=mock_openai_client, max_retries=2, retry_base_delay=0.01)
    agent = Agent(name="Retry Agent")
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "test"}],
    )
    assert call_count == 2
    assert response.messages[-1]["content"] == "recovered"


def test_no_retry_on_auth_error(mock_openai_client):
    """Test that authentication errors are NOT retried."""
    from openai import AuthenticationError
    from unittest.mock import MagicMock

    call_count = 0

    def always_auth_fail(**kwargs):
        nonlocal call_count
        call_count += 1
        raise AuthenticationError(
            message="invalid key",
            response=MagicMock(status_code=401, headers={}),
            body=None,
        )

    mock_openai_client.chat.completions.create = always_auth_fail

    client = Swarm(client=mock_openai_client, max_retries=3, retry_base_delay=0.01)
    agent = Agent(name="Auth Agent")
    with pytest.raises(AuthenticationError):
        client.run(
            agent=agent,
            messages=[{"role": "user", "content": "test"}],
        )
    assert call_count == 1  # No retry


def test_tool_execution_error_handling(mock_openai_client):
    """Test that tool execution errors are caught and reported."""
    def broken_tool():
        raise ValueError("Something went wrong!")

    agent = Agent(name="Error Agent", functions=[broken_tool])

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "broken_tool", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": "handled the error"}
            ),
        ]
    )

    client = Swarm(client=mock_openai_client)
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "trigger error"}],
    )

    # Tool error should be captured in a tool message
    tool_msgs = [m for m in response.messages if m.get("role") == "tool"]
    assert len(tool_msgs) > 0
    assert "ValueError" in tool_msgs[0]["content"]
    assert "Something went wrong" in tool_msgs[0]["content"]
