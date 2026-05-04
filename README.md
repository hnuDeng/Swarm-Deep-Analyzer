# Swarm-Deep-Analyzer: Stateless Multi-Agent Reasoning Engine

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-Swarm-2ea44f)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Experimental-orange)

A highly customized, long-context reasoning system built purely on [OpenAI's Swarm](https://github.com/openai/swarm) framework. This project minimizes local boilerplate code and completely shifts the cognitive load and routing logic to the LLM's native multi-step reasoning capabilities.

---

## Project Abstract

In complex domains such as deep logical auditing and long-document natural language processing (NLP), traditional single-prompt LLM calls frequently suffer from hallucination and "lost-in-the-middle" memory degradation. 

**Swarm-Deep-Analyzer** solves this by orchestrating a specialized network of Agents. Instead of relying on local vector databases (RAG) which chunk and destroy structural integrity, this system utilizes a **Stateless Context Handoff** mechanism. The entire raw context, along with the evolving Chain-of-Thought (CoT) reasoning history, is dynamically passed between expert agents to maintain 100% logical fidelity.

## System Architecture & Multi-Agent Topology

The framework currently implements a three-node sequential reasoning pipeline:

| Agent Node | Core Responsibility | Input | Output / Action |
| :--- | :--- | :--- | :--- |
| **1. Extractor** | Parses raw, massive inputs and extracts structural metadata. | Raw codebases or bulk academic text. | Triggers handoff to Analyzer with structured context. |
| **2. Analyzer** | The core inference engine. Applies rigorous CoT logic to identify flaws or patterns. | Extractor's output + Original Text. | Triggers handoff to Reviewer with a detailed vulnerability/sentiment hypothesis. |
| **3. Reviewer** | Cross-validates the Analyzer's findings against baseline constraints. | Full trajectory from Extractor & Analyzer. | Generates the final structured JSON remediation report. |

## 📈 The Token Consumption Paradigm (Why High API Limits are Required)

This project intentionally sacrifices token economy for reasoning accuracy. Due to Swarm's stateless nature, context window usage grows exponentially during a single execution loop:

* **Step 1:** Extractor processes `T` tokens.
* **Step 2:** Analyzer receives `T` + Extractor's output (`E`), consuming `T + E` tokens.
* **Step 3:** Reviewer receives `T + E` + Analyzer's output (`A`), consuming `T + E + A` tokens.

A standard analysis run involving thousands of lines of code or multi-page academic journals easily consumes **60k - 100k+ tokens per complete cycle**. High rate limits and large token quotas are mathematically essential for the system to function without API throttling.

## Primary Use Cases

### 1. Algorithmic Complexity & Underlying Data Structure Auditing
Traditional static analysis fails on custom data structures. This agent network reads full source code files (C++/Python) dealing with complex structures (e.g., segment trees, disjoint set unions, skew heaps). It traces pointer arithmetic and recursive depth to identify memory management flaws and algorithm complexity bottlenecks that standard tools miss.

### 2. NLP: Long-Document Sentiment Evolution Analysis
The system is optimized for tracking public sentiment shifts across massive datasets (e.g., analyzing regional tourism booms and topic evolution). The Extractor aggregates raw sociological data, the Analyzer maps sentiment trajectories, and the Reviewer compiles the findings into an academic-ready format suitable for SCI/SSCI research methodologies.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hnuDeng/Swarm-Deep-Analyzer.git

## Execution

To initiate the multi-agent pipeline, run:
```bash
python main.py
