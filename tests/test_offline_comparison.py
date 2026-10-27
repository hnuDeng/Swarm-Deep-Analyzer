import json
import os
import subprocess
import sys

import pytest

from tests.mock_client import MockOpenAIClient, create_mock_response


def test_replay_runs_both_paths_with_real_handoffs_and_no_cache(tmp_path, monkeypatch):
    import offline_comparison

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = offline_comparison.run_comparison()
    assert result["mode"] == "deterministic_mock_replay"
    assert len(result["cases"]) == 2
    assert result["cases"][0]["input"] == "def average(total, count):\n    return total / count\n"
    for case in result["cases"]:
        assert case["single"]["stages"] == ["Single Analyzer"]
        assert case["single"]["calls"] == 1
        assert case["multi"]["stages"] == ["Context Extractor", "Deep Logic Analyzer", "Final Reviewer"]
        assert case["multi"]["calls"] == 3
        assert case["single"]["report"]["total_issues"] == len(case["single"]["report"]["issues"])
        assert case["multi"]["report"]["total_issues"] == len(case["multi"]["report"]["issues"])
        assert case["measurements"] == {"accuracy": None, "tokens": None, "cost": None, "latency": None}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("content", ["not json", "{}", '{"issues": [], "summary": "ok", "total_issues": 2}', '{"issues": "wrong", "summary": "ok"}', '{"issues":[{"issue_id":1,"category":"nonsense","severity":"low","confidence":"high","description":"x"}],"summary":"ok"}'])
def test_invalid_final_reports_fail(content):
    import offline_comparison

    client = MockOpenAIClient()
    client.set_response(create_mock_response({"role": "assistant", "content": content}))
    with pytest.raises(ValueError, match="report"):
        offline_comparison.run_path("single", "sample", client)


def test_missing_handoff_does_not_count_as_success():
    import offline_comparison

    client = MockOpenAIClient()
    client.set_response(create_mock_response({"role": "assistant", "content": '{"issues": [], "summary": "premature"}'}))
    with pytest.raises(ValueError, match="stage|handoff"):
        offline_comparison.run_path("multi", "sample", client)


def test_client_error_is_not_reported_as_success():
    import offline_comparison

    client = MockOpenAIClient()
    def fail(**kwargs):
        raise RuntimeError("replay failed")
    client.chat.completions.create = fail
    with pytest.raises(RuntimeError, match="replay failed"):
        offline_comparison.run_path("single", "sample", client)


def test_tool_failure_is_not_reported_as_success():
    import offline_comparison

    client = MockOpenAIClient()
    client.set_response(create_mock_response({"role": "assistant", "content": ""}, [{"name": "transfer_to_analyzer", "args": {"unexpected": 1}}]))
    with pytest.raises(ValueError, match="stage|handoff"):
        offline_comparison.run_path("multi", "sample", client)


def test_accumulated_context_reaches_later_agents():
    import offline_comparison

    cases = json.loads(offline_comparison.CASES_FILE.read_text(encoding="utf-8"))
    case = cases[0]
    client = offline_comparison.ReplayClient(case["replay"]["multi"])
    original = client.chat.completions.create
    requests = []
    def capture(**kwargs):
        requests.append(kwargs["messages"])
        return original(**kwargs)
    client.chat.completions.create = capture
    offline_comparison.run_path("multi", case["input"], client)
    assert all(messages[1]["content"] == case["input"] for messages in requests)
    assert requests[1][2]["content"] == "Synthetic context: average divides by count."
    assert requests[2][4]["content"] == "Synthetic hypothesis: count may be zero."


def test_cli_without_key_or_network():
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    command = "import socket, runpy; socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden')); runpy.run_module('offline_comparison', run_name='__main__')"
    completed = subprocess.run([sys.executable, "-c", command], env=env, capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert result["mode"] == "deterministic_mock_replay"
    assert len(result["cases"]) == 2
