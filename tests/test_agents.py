"""Tests for agent definitions and handoff functions."""
import pytest
from agents import (
    extractor_agent,
    analyzer_agent,
    reviewer_agent,
    transfer_to_analyzer,
    transfer_to_reviewer,
)
from swarm import Agent


def test_extractor_agent_exists():
    assert extractor_agent is not None
    assert isinstance(extractor_agent, Agent)


def test_analyzer_agent_exists():
    assert analyzer_agent is not None
    assert isinstance(analyzer_agent, Agent)


def test_reviewer_agent_exists():
    assert reviewer_agent is not None
    assert isinstance(reviewer_agent, Agent)


def test_extractor_agent_name():
    assert extractor_agent.name == "Context Extractor"


def test_analyzer_agent_name():
    assert analyzer_agent.name == "Deep Logic Analyzer"


def test_reviewer_agent_name():
    assert reviewer_agent.name == "Final Reviewer"


def test_extractor_has_handoff_function():
    func_names = [f.__name__ for f in extractor_agent.functions]
    assert "transfer_to_analyzer" in func_names


def test_analyzer_has_handoff_function():
    func_names = [f.__name__ for f in analyzer_agent.functions]
    assert "transfer_to_reviewer" in func_names


def test_reviewer_has_no_handoff():
    assert len(reviewer_agent.functions) == 0


def test_transfer_to_analyzer_returns_agent():
    result = transfer_to_analyzer()
    assert result is analyzer_agent


def test_transfer_to_reviewer_returns_agent():
    result = transfer_to_reviewer()
    assert result is reviewer_agent


def test_extractor_agent_instructions():
    instructions = extractor_agent.instructions
    assert isinstance(instructions, str)
    assert len(instructions) > 0
    assert "extract" in instructions.lower() or "context" in instructions.lower()


def test_analyzer_agent_instructions():
    instructions = analyzer_agent.instructions
    assert isinstance(instructions, str)
    assert "chain" in instructions.lower() or "reasoning" in instructions.lower() or "cot" in instructions.lower()


def test_reviewer_agent_instructions():
    instructions = reviewer_agent.instructions
    assert isinstance(instructions, str)
    assert "json" in instructions.lower() or "report" in instructions.lower()


def test_all_agents_use_gpt4o():
    """All agents should use gpt-4o as the default model."""
    assert extractor_agent.model == "gpt-4o"
    assert analyzer_agent.model == "gpt-4o"
    assert reviewer_agent.model == "gpt-4o"
