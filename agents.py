"""
Agent definitions for the Swarm-Deep-Analyzer pipeline.

Three-node sequential reasoning topology:
  1. Context Extractor  - Parses raw input, extracts structural metadata
  2. Deep Logic Analyzer - Applies CoT reasoning to identify flaws/patterns
  3. Final Reviewer      - Cross-validates findings, produces JSON report
"""

import os
from swarm import Agent


def load_prompt(filename):
    filepath = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# --- Handoff Functions ---

def transfer_to_analyzer():
    """Hand off extracted context and initial analysis to the deep logic analyzer."""
    return analyzer_agent


def transfer_to_reviewer():
    """Hand off analysis report and original context to the final reviewer."""
    return reviewer_agent


# --- Agent Definitions ---

extractor_agent = Agent(
    name="Context Extractor",
    model="gpt-4o",
    instructions=load_prompt("extractor.md"),
    functions=[transfer_to_analyzer],
)


analyzer_agent = Agent(
    name="Deep Logic Analyzer",
    model="gpt-4o",
    instructions=load_prompt("analyzer.md"),
    functions=[transfer_to_reviewer],
)


reviewer_agent = Agent(
    name="Final Reviewer",
    model="gpt-4o",
    instructions=load_prompt("reviewer.md"),
)


# Export all agents for programmatic access
__all__ = [
    "extractor_agent",
    "analyzer_agent",
    "reviewer_agent",
    "transfer_to_analyzer",
    "transfer_to_reviewer",
]
