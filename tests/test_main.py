"""Tests for main.py entry point."""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock


def test_parse_args_defaults():
    """Test default argument parsing."""
    from main import parse_args

    with patch("sys.argv", ["main.py"]):
        args = parse_args()
    assert args.file is None
    assert args.text is None
    assert args.demo is False
    assert args.debug is False
    assert args.max_turns == 10
    assert args.output is None
    assert args.stream is False


def test_parse_args_with_file():
    """Test argument parsing with file flag."""
    from main import parse_args

    with patch("sys.argv", ["main.py", "--file", "test.txt"]):
        args = parse_args()
    assert args.file == "test.txt"


def test_parse_args_with_text():
    """Test argument parsing with text flag."""
    from main import parse_args

    with patch("sys.argv", ["main.py", "--text", "hello world"]):
        args = parse_args()
    assert args.text == "hello world"


def test_parse_args_with_debug():
    """Test argument parsing with debug flag."""
    from main import parse_args

    with patch("sys.argv", ["main.py", "--debug"]):
        args = parse_args()
    assert args.debug is True


def test_parse_args_with_max_turns():
    """Test argument parsing with max-turns."""
    from main import parse_args

    with patch("sys.argv", ["main.py", "--max-turns", "5"]):
        args = parse_args()
    assert args.max_turns == 5


def test_load_input_with_text():
    """Test load_input with --text argument."""
    from main import load_input, parse_args

    with patch("sys.argv", ["main.py", "--text", "my test input"]):
        args = parse_args()
    result = load_input(args)
    assert result == "my test input"


def test_load_input_with_file(tmp_path):
    """Test load_input with --file argument."""
    from main import load_input, parse_args

    test_file = tmp_path / "test_input.txt"
    test_file.write_text("file content here", encoding="utf-8")

    with patch("sys.argv", ["main.py", "--file", str(test_file)]):
        args = parse_args()
    result = load_input(args)
    assert result == "file content here"


def test_load_input_default():
    """Test load_input with no arguments uses built-in sample."""
    from main import load_input, parse_args, BUILT_IN_SAMPLE

    with patch("sys.argv", ["main.py"]):
        args = parse_args()
    result = load_input(args)
    assert result == BUILT_IN_SAMPLE
    assert len(result) > 100


def test_load_input_missing_file():
    """Test load_input with nonexistent file exits."""
    from main import load_input, parse_args

    with patch("sys.argv", ["main.py", "--file", "nonexistent.txt"]):
        args = parse_args()
    with pytest.raises(SystemExit):
        load_input(args)


def test_built_in_sample_not_empty():
    """The built-in sample should be meaningful."""
    from main import BUILT_IN_SAMPLE
    assert len(BUILT_IN_SAMPLE) > 500
    assert "segment tree" in BUILT_IN_SAMPLE.lower() or "algorithm" in BUILT_IN_SAMPLE.lower()

# --- Batch Mode Tests ---

def test_parse_args_batch():
    from main import parse_args
    with patch("sys.argv", ["main.py", "--batch", "./src", "--pattern", "*.py"]):
        args = parse_args()
    assert args.batch == "./src"
    assert args.pattern == "*.py"


def test_parse_args_batch_defaults():
    from main import parse_args
    with patch("sys.argv", ["main.py", "--batch", "."]):
        args = parse_args()
    assert args.batch == "."
    assert args.pattern == "*.py"
    assert args.max_size == 100000


def test_parse_args_model_override():
    from main import parse_args
    with patch("sys.argv", ["main.py", "--model", "gpt-4o-mini"]):
        args = parse_args()
    assert args.model == "gpt-4o-mini"


def test_parse_args_max_size():
    from main import parse_args
    with patch("sys.argv", ["main.py", "--batch", ".", "--max-size", "50000"]):
        args = parse_args()
    assert args.max_size == 50000


def test_find_batch_files(tmp_path):
    from main import find_batch_files
    # Create test files
    (tmp_path / "a.py").write_text("print('a')", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('b')", encoding="utf-8")
    (tmp_path / "c.txt").write_text("text file", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "d.py").write_text("print('d')", encoding="utf-8")

    files = find_batch_files(str(tmp_path), "*.py", 100000)
    assert len(files) == 3  # a.py, b.py, subdir/d.py
    assert all(f.endswith(".py") for f in files)


def test_find_batch_files_respects_max_size(tmp_path):
    from main import find_batch_files
    (tmp_path / "small.py").write_text("x", encoding="utf-8")
    (tmp_path / "big.py").write_text("x" * 200, encoding="utf-8")

    files = find_batch_files(str(tmp_path), "*.py", 50)
    assert len(files) == 1
    assert "small.py" in files[0]


def test_find_batch_files_empty_dir(tmp_path):
    from main import find_batch_files
    files = find_batch_files(str(tmp_path), "*.py", 100000)
    assert files == []


def test_find_batch_files_nonexistent_dir():
    from main import find_batch_files
    with pytest.raises(SystemExit):
        find_batch_files("/nonexistent/path", "*.py", 100000)


def test_find_batch_files_recursive(tmp_path):
    from main import find_batch_files
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub1" / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "sub1" / "sub2").mkdir()
    (tmp_path / "sub1" / "sub2" / "c.py").write_text("c", encoding="utf-8")

    files = find_batch_files(str(tmp_path), "*.py", 100000)
    assert len(files) == 3
