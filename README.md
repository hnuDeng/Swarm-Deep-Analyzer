<div align="center">

# 🧠 Swarm-Deep-Analyzer

**Stateless Multi-Agent Reasoning Engine**

[English](README.md) | [中文](README_zh.md)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python)
![OpenAI](https://img.shields.io/badge/OpenAI-Swarm-2ea44f?style=for-the-badge&logo=openai)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-Experimental-orange?style=for-the-badge)

A highly customized, long-context reasoning system built purely on [OpenAI's Swarm](https://github.com/openai/swarm) framework. This project minimizes local boilerplate code and completely shifts the cognitive load and routing logic to the LLM's native multi-step reasoning capabilities.

</div>

---

## 📑 Table of Contents

- [📖 Project Abstract](#-project-abstract)
- [✨ Key Features](#-key-features)
- [🏗 System Architecture & Topology](#-system-architecture--topology)
- [⚠️ The Token Consumption Paradigm](#️-the-token-consumption-paradigm)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🚀 Execution](#-execution)
- [💡 Primary Use Cases](#-primary-use-cases)
- [💻 API Usage](#-api-usage)
- [📁 Project Structure](#-project-structure)
- [🧪 Running Tests](#-running-tests)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 📖 Project Abstract

In complex domains such as deep logical auditing and long-document natural language processing (NLP), traditional single-prompt LLM calls frequently suffer from hallucination and "lost-in-the-middle" memory degradation. 

**Swarm-Deep-Analyzer** solves this by orchestrating a specialized network of Agents. Instead of relying on local vector databases (RAG) which chunk and destroy structural integrity, this system utilizes a **Stateless Context Handoff** mechanism. The entire raw context, along with the evolving Chain-of-Thought (CoT) reasoning history, is dynamically passed between expert agents to maintain 100% logical fidelity.

## ✨ Key Features

- **Stateless Context Handoff**: Passes full raw context and CoT history between agents without relying on RAG chunking.
- **Sequential Tri-Node Topology**: `Extractor` ➡️ `Analyzer` ➡️ `Reviewer` for precise logical auditing.
- **High-Fidelity Long-Context Analysis**: Designed for massive text processing, such as full source code or multi-page journals.
- **Batch Processing & Caching**: Analyze entire directories recursively, with built-in caching for efficiency.
- **Extensible Framework**: Easy to define custom agents, handoff rules, and integrate into larger Python applications.

## 🏗 System Architecture & Topology

The framework currently implements a three-node sequential reasoning pipeline:

| Agent Node | Core Responsibility | Input | Output / Action |
| :--- | :--- | :--- | :--- |
| **1. Extractor** | Parses raw, massive inputs and extracts structural metadata. | Raw codebases or bulk academic text. | Triggers handoff to Analyzer with structured context. |
| **2. Analyzer** | The core inference engine. Applies rigorous CoT logic to identify flaws or patterns. | Extractor's output + Original Text. | Triggers handoff to Reviewer with a detailed vulnerability/sentiment hypothesis. |
| **3. Reviewer** | Cross-validates the Analyzer's findings against baseline constraints. | Full trajectory from Extractor & Analyzer. | Generates the final structured JSON remediation report. |

## ⚠️ The Token Consumption Paradigm

> **Note:** Why High API Limits are Required

This project intentionally sacrifices token economy for reasoning accuracy. Due to Swarm's stateless nature, context window usage grows exponentially during a single execution loop:

* **Step 1:** Extractor processes `T` tokens.
* **Step 2:** Analyzer receives `T` + Extractor's output (`E`), consuming `T + E` tokens.
* **Step 3:** Reviewer receives `T + E` + Analyzer's output (`A`), consuming `T + E + A` tokens.

A standard analysis run involving thousands of lines of code or multi-page academic journals easily consumes **60k - 100k+ tokens per complete cycle**. High rate limits and large token quotas are mathematically essential for the system to function without API throttling.

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key with sufficient rate limits

### Install

```bash
# Clone the repository
git clone https://github.com/hnuDeng/Swarm-Deep-Analyzer.git
cd Swarm-Deep-Analyzer

# Install in development mode
pip install -e ".[dev]"
```

### Configure API Key

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# Or on Windows PowerShell
$env:OPENAI_API_KEY = "sk-your-key-here"
```

## 🚀 Execution

### Quick Start (built-in sample)

```bash
python main.py
```

<details>
<summary><strong>More Execution Commands (Click to expand)</strong></summary>

### Analyze a File

```bash
python main.py --file path/to/your/document.txt
```

### Analyze Inline Text

```bash
python main.py --text "Your text to analyze goes here..."
```

### Save Report to File

```bash
python main.py --file input.txt --output report.json
```

### Advanced Modes

```bash
# Enable Debug Logging
python main.py --debug

# Enable Streaming Output
python main.py --stream

# Interactive Demo Loop
python main.py --demo
```
</details>

### 📦 Batch Processing

Analyze multiple files at once:

```bash
# Analyze all Python files in src/ directory
python main.py --batch ./src --pattern "*.py"

# Analyze with custom file size limit
python main.py --batch ./docs --pattern "*.md" --max-size 50000

# Save batch results to directory
python main.py --batch ./src --output ./reports

# Use a cheaper model for batch runs
python main.py --batch ./src --model gpt-4o-mini
```

Batch mode recursively finds matching files, analyzes each one, and optionally saves individual reports plus a summary JSON.

<details>
<summary><strong>All CLI Options</strong></summary>

| Flag | Description | Default |
|------|-------------|---------|
| `--file`, `-f` | Path to input text file | None (uses built-in sample) |
| `--text`, `-t` | Inline text to analyze | None |
| `--batch`, `-b` | Directory path for batch processing | None |
| `--pattern`, `-p` | File pattern for batch mode | `*.py` |
| `--max-size` | Max file size in bytes for batch mode | `100000` |
| `--demo` | Launch interactive REPL | `False` |
| `--debug` | Enable verbose token handoff logging | `False` |
| `--max-turns` | Maximum agent conversation turns | `10` |
| `--output`, `-o` | Save final report to file/directory | None |
| `--stream` | Enable streaming output | `False` |
| `--model` | Override model for all agents | None (uses agent default) |
| `--workers` | Number of concurrent workers for batch processing | `5` |
| `--max-retries`| Max retries for OpenAI API calls | `3` |
| `--cache-dir` | Directory to store cached analysis results | `.swarm_cache` |
| `--no-cache` | Disable caching and force re-analysis | `False` |

</details>

## 💡 Primary Use Cases

### 1. Algorithmic Complexity & Underlying Data Structure Auditing
Traditional static analysis fails on custom data structures. This agent network reads full source code files (C++/Python) dealing with complex structures (e.g., segment trees, disjoint set unions, skew heaps). It traces pointer arithmetic and recursive depth to identify memory management flaws and algorithm complexity bottlenecks that standard tools miss.

### 2. NLP: Long-Document Sentiment Evolution Analysis
The system is optimized for tracking public sentiment shifts across massive datasets (e.g., analyzing regional tourism booms and topic evolution). The Extractor aggregates raw sociological data, the Analyzer maps sentiment trajectories, and the Reviewer compiles the findings into an academic-ready format suitable for SCI/SSCI research methodologies.

## 💻 API Usage

You can also use the framework programmatically:

```python
from swarm import Swarm, Agent
from agents import extractor_agent, analyzer_agent, reviewer_agent

client = Swarm()

# Run the full pipeline
response = client.run(
    agent=extractor_agent,
    messages=[{"role": "user", "content": "Your text here..."}],
    debug=True,
    max_turns=6,
)

print(response.messages[-1]["content"])
```

<details>
<summary><strong>Advanced API Examples</strong></summary>

### Custom Agent Example

```python
from swarm import Swarm, Agent

def transfer_to_expert():
    return expert_agent

triage_agent = Agent(
    name="Triage",
    instructions="Route the user to the right expert.",
    functions=[transfer_to_expert],
)

expert_agent = Agent(
    name="Expert",
    instructions="You are a domain expert. Answer the user's question.",
)

client = Swarm()
response = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "I need help with segment trees."}],
)
```

### Structured Output Parsing

The pipeline output can be parsed into typed Pydantic models:

```python
from swarm import Swarm, Agent, AnalysisReport
from swarm.util import extract_json

client = Swarm()
response = client.run(agent=extractor_agent, messages=[...])
raw = response.messages[-1]["content"]

# Parse JSON from potentially markdown-wrapped output
data = extract_json(raw)
if data:
    report = AnalysisReport(**data)
    print(f"Issues: {report.total_issues}")
    print(f"Critical: {report.critical_count}")
    for issue in report.issues:
        print(f"  [{issue.severity}] {issue.description}")
```

</details>

## 📁 Project Structure

<details>
<summary><strong>View Directory Tree</strong></summary>

```text
Swarm-Deep-Analyzer/
├── main.py                  # Entry point with CLI argument parsing
├── agents.py                # Agent definitions (Extractor, Analyzer, Reviewer)
├── prompts/                 # System prompts for each agent
├── pyproject.toml           # Build configuration and dependencies
├── LICENSE                  # MIT License
├── .gitignore               # Git ignore rules
├── .pre-commit-config.yaml  # Pre-commit hook configuration
├── swarm/                   # Core Swarm framework
│   ├── core.py              # Swarm class (run, streaming, tool handling)
│   ├── types.py             # Agent, Response, Result type definitions
│   └── ...
├── tests/                   # Test suite
└── examples/                # Example applications
```

</details>

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage (if pytest-cov is installed)
pytest --cov=swarm --cov=agents --cov=main
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/hnuDeng/Swarm-Deep-Analyzer/issues). 
If you find this project helpful, please give it a ⭐️!

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---
<div align="center">
Made with ❤️ by <a href="https://github.com/hnuDeng">Deng</a>
</div>
