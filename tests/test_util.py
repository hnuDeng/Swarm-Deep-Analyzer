"""Tests for swarm.util module."""
import pytest
from swarm.util import function_to_json, debug_print, merge_fields, merge_chunk


def test_basic_function():
    def basic_function(arg1, arg2):
        return arg1 + arg2

    result = function_to_json(basic_function)
    assert result == {
        "type": "function",
        "function": {
            "name": "basic_function",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"},
                    "arg2": {"type": "string"},
                },
                "required": ["arg1", "arg2"],
            },
        },
    }


def test_complex_function():
    def complex_function_with_types_and_descriptions(
        arg1: int, arg2: str, arg3: float = 3.14, arg4: bool = False
    ):
        """This is a complex function with a docstring."""
        pass

    result = function_to_json(complex_function_with_types_and_descriptions)
    assert result == {
        "type": "function",
        "function": {
            "name": "complex_function_with_types_and_descriptions",
            "description": "This is a complex function with a docstring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "integer"},
                    "arg2": {"type": "string"},
                    "arg3": {"type": "number"},
                    "arg4": {"type": "boolean"},
                },
                "required": ["arg1", "arg2"],
            },
        },
    }


def test_function_no_args():
    def no_args():
        """Takes no arguments."""
        pass

    result = function_to_json(no_args)
    assert result["function"]["name"] == "no_args"
    assert result["function"]["parameters"]["properties"] == {}
    assert result["function"]["parameters"]["required"] == []


def test_function_with_list_and_dict_types():
    def typed_func(items: list, config: dict):
        pass

    result = function_to_json(typed_func)
    assert result["function"]["parameters"]["properties"]["items"]["type"] == "array"
    assert result["function"]["parameters"]["properties"]["config"]["type"] == "object"


def test_function_with_none_default():
    def func_with_none(x: str = None):
        pass

    result = function_to_json(func_with_none)
    assert "x" in result["function"]["parameters"]["properties"]
    assert "x" not in result["function"]["parameters"]["required"]


def test_function_preserves_docstring():
    def documented():
        """This function has documentation."""
        pass

    result = function_to_json(documented)
    assert result["function"]["description"] == "This function has documentation."


def test_function_no_docstring():
    def undocumented():
        pass

    result = function_to_json(undocumented)
    assert result["function"]["description"] == ""


def test_debug_print_disabled(capsys):
    """debug_print should not output when debug=False."""
    debug_print(False, "should not appear")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_debug_print_enabled(capsys):
    """debug_print should output when debug=True."""
    debug_print(True, "test message")
    captured = capsys.readouterr()
    assert "test message" in captured.out


def test_debug_print_multiple_args(capsys):
    """debug_print should join multiple args."""
    debug_print(True, "arg1", "arg2", "arg3")
    captured = capsys.readouterr()
    assert "arg1 arg2 arg3" in captured.out


def test_merge_fields_string_concatenation():
    target = {"content": "hello"}
    source = {"content": " world"}
    merge_fields(target, source)
    assert target["content"] == "hello world"


def test_merge_fields_nested_dict():
    target = {"a": {"b": "start"}}
    source = {"a": {"b": " end"}}
    merge_fields(target, source)
    assert target["a"]["b"] == "start end"


def test_merge_fields_none_value():
    """None values in source should not overwrite target."""
    target = {"key": "value"}
    source = {"key": None}
    merge_fields(target, source)
    assert target["key"] == "value"


def test_merge_chunk_basic():
    """Test merge_chunk with basic content delta."""
    final_response = {
        "content": "",
        "tool_calls": {},
    }
    delta = {"content": "hello"}
    merge_chunk(final_response, delta)
    assert final_response["content"] == "hello"


def test_merge_chunk_strips_role():
    """merge_chunk should strip 'role' from delta."""
    final_response = {"content": ""}
    delta = {"role": "assistant", "content": "test"}
    merge_chunk(final_response, delta)
    assert "role" not in final_response
    assert final_response["content"] == "test"


def test_function_to_json_output_is_json_serializable():
    """The output of function_to_json should be JSON-serializable."""
    import json

    def sample(a: str, b: int = 0):
        """Sample."""
        pass

    result = function_to_json(sample)
    serialized = json.dumps(result)
    deserialized = json.loads(serialized)
    assert deserialized == result

# --- extract_json Tests ---

def test_extract_json_raw_object():
    from swarm.util import extract_json
    result = extract_json('{"key": "value", "num": 42}')
    assert result == {"key": "value", "num": 42}


def test_extract_json_raw_array():
    from swarm.util import extract_json
    result = extract_json('[1, 2, 3]')
    assert result == [1, 2, 3]


def test_extract_json_markdown_code_block():
    from swarm.util import extract_json
    text = """Here is the report:

```json
{"issues": [{"id": 1, "severity": "high"}]}
```

End of report."""
    result = extract_json(text)
    assert result is not None
    assert result["issues"][0]["id"] == 1


def test_extract_json_code_block_no_lang():
    from swarm.util import extract_json
    text = """Analysis complete:

```
{"summary": "all good"}
```"""
    result = extract_json(text)
    assert result == {"summary": "all good"}


def test_extract_json_embedded_in_prose():
    from swarm.util import extract_json
    text = 'The analysis found {"total_issues": 5, "critical": 2} issues.'
    result = extract_json(text)
    assert result == {"total_issues": 5, "critical": 2}


def test_extract_json_embedded_array():
    from swarm.util import extract_json
    text = 'Results: [{"name": "test"}, {"name": "prod"}] were found.'
    result = extract_json(text)
    assert len(result) == 2
    assert result[0]["name"] == "test"


def test_extract_json_none_on_empty():
    from swarm.util import extract_json
    assert extract_json("") is None
    assert extract_json(None) is None
    assert extract_json("   ") is None


def test_extract_json_none_on_no_json():
    from swarm.util import extract_json
    assert extract_json("This is plain text with no JSON.") is None


def test_extract_json_nested_object():
    from swarm.util import extract_json
    data = '{"report": {"issues": [{"id": 1, "details": {"severity": "high"}}]}}'
    result = extract_json(data)
    assert result["report"]["issues"][0]["details"]["severity"] == "high"


def test_extract_json_with_string_escapes():
    from swarm.util import extract_json
    data = '{"message": "He said \\"hello\\" and left."}'
    result = extract_json(data)
    assert "hello" in result["message"]


def test_extract_json_complex_markdown():
    from swarm.util import extract_json
    text = """# Analysis Report

## Summary
Found several issues.

## JSON Output

```json
{
    "issue_id": 1,
    "category": "logic_flaw",
    "severity": "critical",
    "confidence": "high",
    "location": "main.py:42",
    "description": "Null pointer dereference",
    "recommendation": "Add null check before access"
}
```

## Conclusion
Review the above issues."""
    result = extract_json(text)
    assert result["issue_id"] == 1
    assert result["category"] == "logic_flaw"
    assert result["severity"] == "critical"


def test_extract_json_malformed_returns_none():
    from swarm.util import extract_json
    assert extract_json('{"incomplete":') is None
    assert extract_json('{broken json}') is None
