"""Type definitions for the Swarm framework."""
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from typing import List, Callable, Union, Optional
from enum import Enum

from pydantic import BaseModel, Field, model_validator

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    """Defines an AI agent with its configuration and capabilities."""
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True


class Response(BaseModel):
    """Response from a Swarm run, containing messages and agent state."""
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}


# --- Structured Analysis Report Types ---


class Severity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Confidence(str, Enum):
    """Confidence level for findings."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(str, Enum):
    """Categories of issues that can be identified."""
    LOGIC_FLAW = "logic_flaw"
    COMPLEXITY_ISSUE = "complexity_issue"
    MEMORY_SAFETY = "memory_safety"
    CONCURRENCY = "concurrency"
    DOCUMENTATION_GAP = "documentation_gap"
    DESIGN_WEAKNESS = "design_weakness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    OTHER = "other"


class Issue(BaseModel):
    """A single issue identified during analysis."""
    issue_id: int = Field(description="Sequential identifier")
    category: IssueCategory = Field(description="Category of the issue")
    severity: Severity = Field(description="Severity level")
    confidence: Confidence = Field(description="Confidence in this finding")
    location: str = Field(default="", description="Where the issue was found")
    description: str = Field(description="Detailed explanation")
    recommendation: str = Field(default="", description="Suggested fix")

    @model_validator(mode="before")
    @classmethod
    def coerce_category(cls, data):
        """Accept string category values that match enum values."""
        if isinstance(data, dict) and "category" in data:
            cat = data["category"]
            if isinstance(cat, str):
                # Normalize: replace spaces with underscores, lowercase
                normalized = cat.strip().lower().replace(" ", "_").replace("-", "_")
                valid_values = [e.value for e in IssueCategory]
                if normalized not in valid_values:
                    data["category"] = IssueCategory.OTHER.value
                else:
                    data["category"] = normalized
        return data


class AnalysisReport(BaseModel):
    """Structured output from the analysis pipeline."""
    issues: List[Issue] = Field(default_factory=list, description="List of identified issues")
    summary: str = Field(default="", description="Overall assessment summary")
    total_issues: Optional[int] = Field(default=None, description="Total number of issues")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    @model_validator(mode="after")
    def compute_total(self):
        """Auto-compute total_issues if not provided."""
        if self.total_issues is None:
            self.total_issues = len(self.issues)
        return self

    @property
    def critical_count(self) -> int:
        """Count of critical severity issues."""
        return sum(1 for i in self.issues if i.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high severity issues."""
        return sum(1 for i in self.issues if i.severity == Severity.HIGH)

    def issues_by_category(self) -> dict:
        """Group issues by category."""
        result = {}
        for issue in self.issues:
            cat = issue.category.value if isinstance(issue.category, IssueCategory) else issue.category
            result.setdefault(cat, []).append(issue)
        return result

    def issues_by_severity(self) -> dict:
        """Group issues by severity."""
        result = {}
        for issue in self.issues:
            sev = issue.severity.value if isinstance(issue.severity, Severity) else issue.severity
            result.setdefault(sev, []).append(issue)
        return result


__all__ = [
    "Agent", "Response", "Result", "AgentFunction",
    "ChatCompletionMessage", "ChatCompletionMessageToolCall", "Function",
    "AnalysisReport", "Issue", "Severity", "Confidence", "IssueCategory",
]
