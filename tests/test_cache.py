import json
from unittest.mock import patch

import pytest

from agents import extractor_agent, reviewer_agent
from main import run_deep_analysis
from swarm.types import Response


@pytest.fixture
def analysis_client():
    with patch("main.Swarm") as client_factory:
        client = client_factory.return_value
        client.run.return_value = Response(
            messages=[{"role": "assistant", "content": "first report"}],
            agent=reviewer_agent,
            context_variables={"source": "test"},
        )
        yield client


def test_cached_response_preserves_messages_and_context(tmp_path, analysis_client):
    first = run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)
    cached = run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)

    assert isinstance(cached, Response)
    assert cached.messages == first.messages
    assert cached.context_variables == {"source": "test"}
    assert cached.agent is reviewer_agent
    assert analysis_client.run.call_count == 1


@pytest.mark.parametrize("changed_options", [
    {"model_override": "another-model"},
    {"max_turns": 2},
])
def test_cache_separates_analysis_settings(tmp_path, analysis_client, changed_options):
    run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)
    analysis_client.run.return_value = Response(
        messages=[{"role": "assistant", "content": "new settings report"}],
    )

    response = run_deep_analysis(
        "same input", cache_dir=tmp_path, quiet=True, **changed_options,
    )

    assert response.messages[-1]["content"] == "new settings report"
    assert analysis_client.run.call_count == 2


def test_cache_invalidates_when_prompt_changes(tmp_path, analysis_client, monkeypatch):
    run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)
    monkeypatch.setattr(extractor_agent, "instructions", "Updated extraction rules")
    analysis_client.run.return_value = Response(
        messages=[{"role": "assistant", "content": "updated prompt report"}],
    )

    response = run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)

    assert response.messages[-1]["content"] == "updated prompt report"
    assert analysis_client.run.call_count == 2


def test_corrupt_cache_is_replaced(tmp_path, analysis_client):
    run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)
    cache_file = next(tmp_path.glob("*.json"))
    cache_file.write_text("{broken", encoding="utf-8")
    analysis_client.run.return_value = Response(
        messages=[{"role": "assistant", "content": "recovered report"}],
    )

    response = run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)

    assert response.messages[-1]["content"] == "recovered report"
    assert json.loads(cache_file.read_text(encoding="utf-8"))
    assert analysis_client.run.call_count == 2


def test_no_cache_ignores_existing_result(tmp_path, analysis_client):
    run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)
    analysis_client.run.return_value = Response(
        messages=[{"role": "assistant", "content": "fresh report"}],
    )

    response = run_deep_analysis(
        "same input", cache_dir=tmp_path, quiet=True, no_cache=True,
    )

    assert response.messages[-1]["content"] == "fresh report"
    assert analysis_client.run.call_count == 2


def test_streaming_result_can_be_loaded_from_cache(tmp_path, analysis_client):
    final_response = analysis_client.run.return_value
    analysis_client.run.return_value = iter([
        {"content": "first report"},
        {"response": final_response},
    ])
    run_deep_analysis("same input", cache_dir=tmp_path, quiet=True, stream=True)

    cached = run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)

    assert isinstance(cached, Response)
    assert cached.context_variables == {"source": "test"}
    assert cached.messages[-1]["content"] == "first report"
    assert analysis_client.run.call_count == 1


def test_unavailable_cache_does_not_prevent_analysis(tmp_path, analysis_client):
    cache_path = tmp_path / "not-a-directory"
    cache_path.write_text("occupied", encoding="utf-8")

    response = run_deep_analysis("same input", cache_dir=cache_path, quiet=True)

    assert response.messages[-1]["content"] == "first report"
    assert cache_path.read_text(encoding="utf-8") == "occupied"


def test_empty_response_does_not_crash_cache_write(tmp_path, analysis_client):
    analysis_client.run.return_value = Response()

    response = run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)

    assert response.messages == []
    assert not list(tmp_path.glob("*.json"))


def test_dynamic_prompts_bypass_cache(tmp_path, analysis_client, monkeypatch):
    monkeypatch.setattr(extractor_agent, "instructions", lambda context: "Dynamic rules")
    run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)
    analysis_client.run.return_value = Response(
        messages=[{"role": "assistant", "content": "dynamic prompt report"}],
    )

    response = run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)

    assert response.messages[-1]["content"] == "dynamic prompt report"
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("messages", [[None], ["not a message"], [{}]])
def test_malformed_cached_messages_trigger_analysis(tmp_path, analysis_client, messages):
    run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)
    cache_file = next(tmp_path.glob("*.json"))
    cache_file.write_text(json.dumps({
        "messages": messages, "context_variables": {}, "agent_name": None,
    }), encoding="utf-8")
    analysis_client.run.return_value = Response(
        messages=[{"role": "assistant", "content": "recovered report"}],
    )

    response = run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)

    assert response.messages[-1]["content"] == "recovered report"
    assert analysis_client.run.call_count == 2


def test_cache_cleanup_failure_does_not_hide_analysis(tmp_path, analysis_client):
    with patch("main.os.replace", side_effect=PermissionError("cache locked")), patch(
        "main.Path.unlink", side_effect=PermissionError("temporary file locked"),
    ):
        response = run_deep_analysis("same input", cache_dir=tmp_path, quiet=True)

    assert response.messages[-1]["content"] == "first report"
