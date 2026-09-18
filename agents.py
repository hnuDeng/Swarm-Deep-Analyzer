"""
Agent definitions for the Swarm-Deep-Analyzer pipeline.

Three-node sequential reasoning topology:
  1. Context Extractor  - Parses raw input, extracts structural metadata
  2. Deep Logic Analyzer - Applies CoT reasoning to identify flaws/patterns
  3. Final Reviewer      - Cross-validates findings, produces JSON report
"""

from swarm import Agent


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
    instructions="""You are a top-tier context extraction specialist.
Your task is to read the user's input material (which may be tens of pages
of academic literature, large codebases, or structured documents).

Extract the following:
1. Core logical architecture or structural overview
2. Key data structures and their relationships
3. Algorithm complexity bottlenecks
4. Core arguments or thesis points (for academic text)
5. Potential areas of concern or weakness

After extraction, you MUST call transfer_to_analyzer to hand off the full
context along with your structured extraction notes.""",
    functions=[transfer_to_analyzer],
)


analyzer_agent = Agent(
    name="Deep Logic Analyzer",
    model="gpt-4o",
    instructions="""You are a deep logic analyst. You have received the full
context history from the Context Extractor.

Apply multi-step Chain-of-Thought (CoT) reasoning to:
1. Identify logical flaws, race conditions, or memory safety issues in code
2. Trace pointer arithmetic and recursive depth for memory management problems
3. Detect sentiment contradictions or argumentative weaknesses in text
4. Assess algorithm complexity claims vs actual implementation behavior
5. Map temporal evolution of themes or patterns across long documents

Since the context is extremely long, maintain strict logical rigor throughout.
After completing your analysis, call transfer_to_reviewer to hand off your
findings along with the full trajectory.""",
    functions=[transfer_to_reviewer],
)


reviewer_agent = Agent(
    name="Final Reviewer",
    model="gpt-4o",
    instructions="""You are the final reviewer. Carefully read the entire
interaction history from all previous agents.

Your responsibilities:
1. Cross-validate the Analyzer's findings against baseline constraints
2. Verify that each identified issue is grounded in the actual source material
3. Assess severity and confidence level for each finding
4. Generate a structured JSON remediation report with:
   - issue_id: sequential identifier
   - category: one of [logic_flaw, complexity_issue, memory_safety,
     concurrency, documentation_gap, design_weakness]
   - severity: one of [critical, high, medium, low]
   - confidence: one of [high, medium, low]
   - location: where in the source the issue was found
   - description: detailed explanation
   - recommendation: specific fix or improvement suggestion
5. Provide an overall assessment summary

Output your final report as a well-formatted JSON object.""",
)


# Export all agents for programmatic access
__all__ = [
    "extractor_agent",
    "analyzer_agent",
    "reviewer_agent",
    "transfer_to_analyzer",
    "transfer_to_reviewer",
]
