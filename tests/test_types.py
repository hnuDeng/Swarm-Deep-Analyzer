"""Tests for structured analysis report types."""
import pytest
from swarm.types import (
    AnalysisReport, Issue, Severity, Confidence, IssueCategory,
)


def test_issue_creation():
    issue = Issue(
        issue_id=1,
        category="logic_flaw",
        severity="critical",
        confidence="high",
        location="main.py:42",
        description="Null pointer dereference",
        recommendation="Add null check",
    )
    assert issue.issue_id == 1
    assert issue.category == IssueCategory.LOGIC_FLAW
    assert issue.severity == Severity.CRITICAL
    assert issue.confidence == Confidence.HIGH


def test_issue_category_coercion():
    """Unknown categories should be coerced to 'other'."""
    issue = Issue(
        issue_id=1,
        category="some_unknown_category",
        severity="low",
        confidence="low",
        description="test",
    )
    assert issue.category == IssueCategory.OTHER


def test_issue_category_normalization():
    """Category with spaces/hyphens should be normalized."""
    issue = Issue(
        issue_id=1,
        category="memory-safety",
        severity="high",
        confidence="medium",
        description="test",
    )
    assert issue.category == IssueCategory.MEMORY_SAFETY


def test_analysis_report_creation():
    report = AnalysisReport(
        issues=[
            Issue(issue_id=1, category="logic_flaw", severity="critical", confidence="high", description="bug"),
            Issue(issue_id=2, category="performance", severity="low", confidence="medium", description="slow"),
        ],
        summary="Found 2 issues",
    )
    assert report.total_issues == 2
    assert len(report.issues) == 2
    assert report.summary == "Found 2 issues"


def test_analysis_report_auto_total():
    """total_issues should auto-compute from issues list."""
    report = AnalysisReport(
        issues=[
            Issue(issue_id=1, category="logic_flaw", severity="critical", confidence="high", description="a"),
            Issue(issue_id=2, category="logic_flaw", severity="high", confidence="high", description="b"),
            Issue(issue_id=3, category="logic_flaw", severity="medium", confidence="high", description="c"),
        ],
        summary="test",
    )
    assert report.total_issues == 3


def test_analysis_report_critical_count():
    report = AnalysisReport(
        issues=[
            Issue(issue_id=1, category="logic_flaw", severity="critical", confidence="high", description="a"),
            Issue(issue_id=2, category="logic_flaw", severity="critical", confidence="high", description="b"),
            Issue(issue_id=3, category="logic_flaw", severity="low", confidence="high", description="c"),
        ],
        summary="test",
    )
    assert report.critical_count == 2
    assert report.high_count == 0


def test_analysis_report_by_category():
    report = AnalysisReport(
        issues=[
            Issue(issue_id=1, category="logic_flaw", severity="critical", confidence="high", description="a"),
            Issue(issue_id=2, category="performance", severity="low", confidence="medium", description="b"),
            Issue(issue_id=3, category="logic_flaw", severity="medium", confidence="high", description="c"),
        ],
        summary="test",
    )
    by_cat = report.issues_by_category()
    assert "logic_flaw" in by_cat
    assert len(by_cat["logic_flaw"]) == 2
    assert "performance" in by_cat
    assert len(by_cat["performance"]) == 1


def test_analysis_report_by_severity():
    report = AnalysisReport(
        issues=[
            Issue(issue_id=1, category="logic_flaw", severity="critical", confidence="high", description="a"),
            Issue(issue_id=2, category="logic_flaw", severity="low", confidence="medium", description="b"),
            Issue(issue_id=3, category="logic_flaw", severity="critical", confidence="high", description="c"),
        ],
        summary="test",
    )
    by_sev = report.issues_by_severity()
    assert len(by_sev["critical"]) == 2
    assert len(by_sev["low"]) == 1


def test_analysis_report_empty():
    report = AnalysisReport(summary="No issues found")
    assert report.total_issues == 0
    assert report.critical_count == 0
    assert report.issues_by_category() == {}


def test_analysis_report_from_dict():
    """Test creating a report from a raw dict (as would come from JSON parsing)."""
    data = {
        "issues": [
            {
                "issue_id": 1,
                "category": "memory_safety",
                "severity": "critical",
                "confidence": "high",
                "location": "core.py:100",
                "description": "Buffer overflow",
                "recommendation": "Use bounds checking",
            }
        ],
        "summary": "One critical issue found",
    }
    report = AnalysisReport(**data)
    assert report.issues[0].category == IssueCategory.MEMORY_SAFETY
    assert report.issues[0].severity == Severity.CRITICAL
    assert report.total_issues == 1


def test_analysis_report_json_roundtrip():
    """Test that report can be serialized and deserialized."""
    import json
    report = AnalysisReport(
        issues=[
            Issue(issue_id=1, category="logic_flaw", severity="critical", confidence="high", description="test"),
        ],
        summary="roundtrip test",
    )
    serialized = report.model_dump_json()
    deserialized = AnalysisReport.model_validate_json(serialized)
    assert deserialized.total_issues == report.total_issues
    assert deserialized.issues[0].description == "test"


def test_severity_enum_values():
    assert Severity.CRITICAL.value == "critical"
    assert Severity.HIGH.value == "high"
    assert Severity.MEDIUM.value == "medium"
    assert Severity.LOW.value == "low"


def test_issue_category_enum_values():
    assert IssueCategory.LOGIC_FLAW.value == "logic_flaw"
    assert IssueCategory.MEMORY_SAFETY.value == "memory_safety"
    assert IssueCategory.CONCURRENCY.value == "concurrency"
    assert IssueCategory.SECURITY.value == "security"


def test_analysis_report_with_metadata():
    report = AnalysisReport(
        issues=[],
        summary="test",
        metadata={"input_tokens": 50000, "model": "gpt-4o", "duration_seconds": 12.5},
    )
    assert report.metadata["input_tokens"] == 50000
    assert report.metadata["model"] == "gpt-4o"
