"""Utility functions for the Swarm framework."""
import inspect
import json
import re
from datetime import datetime
from typing import Any, Optional


def debug_print(debug: bool, *args: str) -> None:
    """Print debug information with timestamp when debug mode is enabled."""
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target, source):
    """Recursively merge source dict into target, concatenating strings."""
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    """Merge a streaming delta chunk into the accumulated response."""
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def extract_json(text: str) -> Optional[Any]:
    """
    Extract and parse JSON from text that may contain markdown code blocks,
    prose, or other non-JSON content.

    Handles:
    - Raw JSON strings
    - JSON inside ```json ... ``` code blocks
    - JSON inside ``` ... ``` code blocks (no language tag)
    - Text with embedded JSON objects/arrays

    Returns:
        Parsed JSON data, or None if no valid JSON found.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try direct parse first (fastest path)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except (json.JSONDecodeError, ValueError):
            continue

    # Try finding the outermost JSON object or array
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        start = text.find(start_char)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            c = text[i]

            if escape_next:
                escape_next = False
                continue

            if c == '\\' and in_string:
                escape_next = True
                continue

            if c == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break

    return None


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
