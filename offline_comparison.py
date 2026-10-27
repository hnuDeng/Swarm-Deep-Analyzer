"""Deterministic local replay of single and multi-agent Swarm paths."""

import json
from pathlib import Path
from types import SimpleNamespace

from openai.types.chat.chat_completion import ChatCompletion, Choice

from agents import extractor_agent
from swarm import Agent, Swarm
from swarm.types import AnalysisReport, ChatCompletionMessage, ChatCompletionMessageToolCall, Function, IssueCategory


CASES_FILE = Path(__file__).with_name("offline_cases.json")
SINGLE = Agent(name="Single Analyzer", instructions="Analyze the input and return a structured JSON report.")


class ReplayCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)

    def create(self, **kwargs):
        try:
            item = next(self.responses)
        except StopIteration as exc:
            raise RuntimeError("Replay exhausted before Swarm finished") from exc
        calls = [ChatCompletionMessageToolCall(
            id=f"replay-{index}", type="function",
            function=Function(name=name, arguments="{}"),
        ) for index, name in enumerate(item.get("tools", []))]
        message = ChatCompletionMessage(role="assistant", content=item["content"], tool_calls=calls or None)
        return ChatCompletion(id="offline-replay", created=0, model="offline-replay", object="chat.completion",
                              choices=[Choice(index=0, message=message, finish_reason="tool_calls" if calls else "stop")])


class ReplayClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=ReplayCompletions(responses))


class CountingClient:
    def __init__(self, client):
        self.client = client
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        return self.client.chat.completions.create(**kwargs)


def validate_report(content):
    try:
        data = json.loads(content)
        if not isinstance(data, dict) or not isinstance(data.get("issues"), list) or not isinstance(data.get("summary"), str) or not data["summary"].strip():
            raise ValueError("issues and nonempty summary are required")
        for issue in data["issues"]:
            category = issue.get("category") if isinstance(issue, dict) else None
            if not isinstance(category, str) or category.strip().lower().replace(" ", "_").replace("-", "_") not in {item.value for item in IssueCategory}:
                raise ValueError("invalid issue category")
        report = AnalysisReport.model_validate(data)
        if report.total_issues != len(report.issues):
            raise ValueError("total_issues does not match issues")
        return report.model_dump(mode="json")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid report: {exc}") from exc


def run_path(mode, input_text, client):
    if mode not in ("single", "multi"):
        raise ValueError(f"Unknown path: {mode}")
    counter = CountingClient(client)
    response = Swarm(client=counter, max_retries=0).run(
        agent=SINGLE if mode == "single" else extractor_agent,
        messages=[{"role": "user", "content": input_text}], max_turns=10,
    )
    assistants = [message for message in response.messages if message.get("role") == "assistant"]
    stages = [message.get("sender") for message in assistants]
    expected = ["Single Analyzer"] if mode == "single" else ["Context Extractor", "Deep Logic Analyzer", "Final Reviewer"]
    if stages != expected or counter.calls != len(expected) or response.agent is None or response.agent.name != expected[-1]:
        raise ValueError(f"Invalid stage sequence or handoff: {stages}")
    tool_messages = [message for message in response.messages if message.get("role") == "tool"]
    expected_tools = [] if mode == "single" else ["transfer_to_analyzer", "transfer_to_reviewer"]
    expected_targets = [] if mode == "single" else ["Deep Logic Analyzer", "Final Reviewer"]
    if [message.get("tool_name") for message in tool_messages] != expected_tools:
        raise ValueError("Invalid handoff sequence")
    for message, target in zip(tool_messages, expected_targets):
        try:
            value = json.loads(message.get("content", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid handoff result") from exc
        if value != {"assistant": target}:
            raise ValueError("Invalid handoff result")
    return {"stages": stages, "calls": counter.calls, "report": validate_report(assistants[-1].get("content"))}


def run_comparison():
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        result = {"id": case["id"], "input": case["input"], "human_label": case["human_label"]}
        for mode in ("single", "multi"):
            result[mode] = run_path(mode, case["input"], ReplayClient(case["replay"][mode]))
        result["measurements"] = {"accuracy": None, "tokens": None, "cost": None, "latency": None}
        results.append(result)
    return {"mode": "deterministic_mock_replay", "limitation": "Static replay validates the harness only, not actual LLM quality.", "cases": results}


if __name__ == "__main__":
    print(json.dumps(run_comparison(), ensure_ascii=False, indent=2))
