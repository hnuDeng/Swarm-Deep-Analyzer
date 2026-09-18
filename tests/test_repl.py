"""Tests for swarm.repl module."""
import pytest
import json
from swarm.repl.repl import (
    process_and_print_streaming_response,
    pretty_print_messages,
)
from swarm.types import Response, Agent


def test_pretty_print_messages_assistant(capsys):
    """Test pretty_print_messages with assistant messages."""
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "sender": "TestAgent",
            "content": "Hello! How can I help?",
            "tool_calls": None,
        },
    ]
    pretty_print_messages(messages)
    captured = capsys.readouterr()
    assert "TestAgent" in captured.out
    assert "Hello! How can I help?" in captured.out


def test_pretty_print_messages_skips_user(capsys):
    """pretty_print_messages should skip non-assistant messages."""
    messages = [
        {"role": "user", "content": "hello"},
    ]
    pretty_print_messages(messages)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_pretty_print_messages_with_tool_calls(capsys):
    """Test pretty_print_messages with tool call messages."""
    messages = [
        {
            "role": "assistant",
            "sender": "TestAgent",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "NYC"}),
                    }
                }
            ],
        },
    ]
    pretty_print_messages(messages)
    captured = capsys.readouterr()
    assert "get_weather" in captured.out


def test_pretty_print_messages_empty(capsys):
    """Test pretty_print_messages with empty list."""
    pretty_print_messages([])
    captured = capsys.readouterr()
    assert captured.out == ""


def test_pretty_print_messages_multiple_tool_calls(capsys):
    """Test pretty_print_messages with multiple tool calls."""
    messages = [
        {
            "role": "assistant",
            "sender": "TestAgent",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "func_a",
                        "arguments": json.dumps({"x": 1}),
                    }
                },
                {
                    "function": {
                        "name": "func_b",
                        "arguments": json.dumps({"y": 2}),
                    }
                },
            ],
        },
    ]
    pretty_print_messages(messages)
    captured = capsys.readouterr()
    assert "func_a" in captured.out
    assert "func_b" in captured.out

# --- Streaming Response Tests ---

def test_process_streaming_response_basic(capsys):
    """Test process_and_print_streaming_response with basic content chunks."""
    from swarm.types import Response

    chunks = [
        {"sender": "TestAgent"},
        {"content": "Hello"},
        {"content": " world"},
        {"delim": "end"},
        {"response": Response(messages=[], agent=None, context_variables={})},
    ]

    result = process_and_print_streaming_response(iter(chunks))
    captured = capsys.readouterr()
    assert "Hello" in captured.out
    assert "world" in captured.out
    assert isinstance(result, Response)


def test_process_streaming_response_returns_response(capsys):
    """Test that process_and_print_streaming_response returns the final Response."""
    from swarm.types import Response

    mock_response = Response(
        messages=[{"role": "assistant", "content": "test"}],
        agent=None,
        context_variables={},
    )
    chunks = [
        {"content": "test"},
        {"response": mock_response},
    ]

    result = process_and_print_streaming_response(iter(chunks))
    assert result is mock_response


def test_process_streaming_response_no_content(capsys):
    """Test process_and_print_streaming_response with no content."""
    chunks = [
        {"delim": "start"},
        {"delim": "end"},
    ]

    result = process_and_print_streaming_response(iter(chunks))
    captured = capsys.readouterr()
    # No newlines printed for empty content
    assert result is None


def test_process_streaming_response_tool_calls(capsys):
    """Test process_and_print_streaming_response with tool calls."""
    chunks = [
        {"sender": "Agent"},
        {"tool_calls": [{"function": {"name": "my_tool"}}]},
    ]

    process_and_print_streaming_response(iter(chunks))
    captured = capsys.readouterr()
    assert "my_tool" in captured.out


def test_process_streaming_response_skips_empty_tool_names(capsys):
    """Test that empty tool call names are skipped."""
    chunks = [
        {"sender": "Agent"},
        {"tool_calls": [{"function": {"name": ""}}]},
    ]

    process_and_print_streaming_response(iter(chunks))
    captured = capsys.readouterr()
    # Empty name should not produce output
    assert "()" not in captured.out or captured.out.strip() == ""


def test_process_streaming_response_multiple_messages(capsys):
    """Test process_and_print_streaming_response with multiple message boundaries."""
    from swarm.types import Response

    chunks = [
        {"sender": "Agent1"},
        {"content": "first"},
        {"delim": "end"},
        {"sender": "Agent2"},
        {"content": "second"},
        {"delim": "end"},
        {"response": Response(messages=[], agent=None, context_variables={})},
    ]

    result = process_and_print_streaming_response(iter(chunks))
    captured = capsys.readouterr()
    assert "first" in captured.out
    assert "second" in captured.out
