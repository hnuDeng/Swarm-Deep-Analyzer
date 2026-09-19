<div align="center">

# 🧠 Swarm-Deep-Analyzer

**无状态多智能体推理引擎**

[English](README.md) | [中文](README_zh.md)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python)
![OpenAI](https://img.shields.io/badge/OpenAI-Swarm-2ea44f?style=for-the-badge&logo=openai)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-Experimental-orange?style=for-the-badge)

一个完全基于 [OpenAI Swarm](https://github.com/openai/swarm) 框架构建的高度定制化的长上下文推理系统。该项目最大限度地减少了本地样板代码，将认知负载和路由逻辑完全交给了 LLM 原生的多步推理能力。

</div>

---

## 📑 目录

- [📖 项目简介](#-项目简介)
- [✨ 核心特性](#-核心特性)
- [🏗 系统架构与拓扑](#-系统架构与拓扑)
- [⚠️ Token 消耗范式](#️-token-消耗范式)
- [⚙️ 安装与配置](#️-安装与配置)
- [🚀 执行方式](#-执行方式)
- [💡 主要用例](#-主要用例)
- [💻 API 使用方式](#-api-使用方式)
- [📁 项目结构](#-项目结构)
- [🧪 运行测试](#-运行测试)
- [🤝 贡献](#-贡献)
- [📄 许可证](#-许可证)

---

## 📖 项目简介

在诸如深度逻辑审计和长文档自然语言处理（NLP）等复杂领域，传统的单提示词 LLM 调用经常面临幻觉和“迷失在中间”的记忆退化问题。

**Swarm-Deep-Analyzer** 通过编排一个专门的智能体（Agents）网络来解决这个问题。该系统不依赖于对数据分块从而破坏结构完整性的本地向量数据库（RAG），而是采用了**无状态上下文交接（Stateless Context Handoff）**机制。完整的原始上下文连同不断演进的思维链（CoT）推理历史，在专家智能体之间动态传递，以保持 100% 的逻辑保真度。

## ✨ 核心特性

- **无状态上下文交接**：在智能体之间传递完整的原始上下文和思维链历史，无需依赖 RAG 分块。
- **顺次三节点拓扑**：`提取器 (Extractor)` ➡️ `分析器 (Analyzer)` ➡️ `审查器 (Reviewer)`，实现精确的逻辑审计。
- **高保真长上下文分析**：专为海量文本处理设计，如完整的源代码或多页学术期刊。
- **批处理与缓存**：递归分析整个目录，内置缓存机制以提升效率。
- **可扩展框架**：轻松定义自定义智能体、交接规则，并可集成到大型 Python 应用中。

## 🏗 系统架构与拓扑

该框架目前实现了一个三节点顺序推理流水线：

| 智能体节点 | 核心职责 | 输入 | 输出 / 动作 |
| :--- | :--- | :--- | :--- |
| **1. 提取器 (Extractor)** | 解析庞大的原始输入并提取结构化元数据。 | 源代码库或海量学术文本。 | 携带结构化上下文触发至分析器的交接。 |
| **2. 分析器 (Analyzer)** | 核心推理引擎。应用严谨的 CoT 逻辑识别缺陷或模式。 | 提取器的输出 + 原始文本。 | 携带详细的漏洞/情感假设触发至审查器的交接。 |
| **3. 审查器 (Reviewer)** | 根据基线约束交叉验证分析器的发现。 | 提取器和分析器的完整轨迹。 | 生成最终的结构化 JSON 修复报告。 |

## ⚠️ Token 消耗范式

> **注意：** 为什么需要高 API 限制

该项目有意牺牲 Token 经济性以换取推理准确性。由于 Swarm 的无状态特性，在一个执行循环中上下文窗口的使用量会呈指数级增长：

* **第 1 步：** 提取器处理 `T` 个 Token。
* **第 2 步：** 分析器接收 `T` + 提取器的输出 (`E`)，消耗 `T + E` 个 Token。
* **第 3 步：** 审查器接收 `T + E` + 分析器的输出 (`A`)，消耗 `T + E + A` 个 Token。

一次标准的分析（涉及数千行代码或多页学术期刊）在一个完整循环中很容易消耗 **60k - 100k+ Token**。因此，高并发限制和大量 Token 额度在数学上是系统无 API 节流运行的必要条件。

## ⚙️ 安装与配置

### 前置要求

- Python 3.10 或更高版本
- 一个具有足够限额的 OpenAI API 密钥

### 安装

```bash
# 克隆仓库
git clone https://github.com/hnuDeng/Swarm-Deep-Analyzer.git
cd Swarm-Deep-Analyzer

# 以开发模式安装
pip install -e ".[dev]"
```

### 配置 API 密钥

```bash
# 设置你的 OpenAI API 密钥
export OPENAI_API_KEY="sk-your-key-here"

# 或者在 Windows PowerShell 下
$env:OPENAI_API_KEY = "sk-your-key-here"
```

## 🚀 执行方式

### 快速开始 (内置示例)

```bash
python main.py
```

<details>
<summary><strong>更多执行命令 (点击展开)</strong></summary>

### 分析文件

```bash
python main.py --file path/to/your/document.txt
```

### 分析内联文本

```bash
python main.py --text "需要分析的文本..."
```

### 将报告保存到文件

```bash
python main.py --file input.txt --output report.json
```

### 高级模式

```bash
# 启用 Debug 日志
python main.py --debug

# 启用流式输出
python main.py --stream

# 启动交互式演示循环
python main.py --demo
```
</details>

### 📦 批处理

一次性分析多个文件：

```bash
# 分析 src/ 目录下的所有 Python 文件
python main.py --batch ./src --pattern "*.py"

# 使用自定义文件大小限制分析
python main.py --batch ./docs --pattern "*.md" --max-size 50000

# 将批处理结果保存到目录
python main.py --batch ./src --output ./reports

# 对于批处理任务使用更便宜的模型
python main.py --batch ./src --model gpt-4o-mini
```

批处理模式会递归查找匹配的文件，分析每个文件，并可选择保存单独的报告以及一个汇总的 JSON 文件。

<details>
<summary><strong>全部 CLI 选项</strong></summary>

| 参数 | 描述 | 默认值 |
|------|-------------|---------|
| `--file`, `-f` | 输入文本文件的路径 | None (使用内置示例) |
| `--text`, `-t` | 要分析的内联文本 | None |
| `--batch`, `-b` | 批处理的目录路径 | None |
| `--pattern`, `-p` | 批处理模式下的文件匹配模式 | `*.py` |
| `--max-size` | 批处理模式下的最大文件大小（字节）| `100000` |
| `--demo` | 启动交互式 REPL | `False` |
| `--debug` | 启用详细的 Token 交接日志 | `False` |
| `--max-turns` | 智能体对话的最大轮数 | `10` |
| `--output`, `-o` | 将最终报告保存到文件/目录 | None |
| `--stream` | 启用流式输出 | `False` |
| `--model` | 覆盖所有智能体的模型 | None (使用智能体默认值) |
| `--workers` | 批处理的并发工作线程数 | `5` |
| `--max-retries`| OpenAI API 调用的最大重试次数 | `3` |
| `--cache-dir` | 存储分析结果缓存的目录 | `.swarm_cache` |
| `--no-cache` | 禁用缓存并强制重新分析 | `False` |

</details>

## 💡 主要用例

### 1. 算法复杂度与底层数据结构审计
传统的静态分析在自定义数据结构上往往会失效。该智能体网络能够阅读涉及复杂结构（例如，线段树、并查集、斜堆）的完整源代码文件（C++/Python）。它能追踪指针运算和递归深度，以识别标准工具遗漏的内存管理缺陷和算法复杂度瓶颈。

### 2. NLP: 长文档情感演变分析
系统经过优化，能够跨海量数据集追踪公众情感的变化（例如，分析区域旅游热潮和话题演变）。提取器汇总原始社会学数据，分析器绘制情感轨迹，审查器将发现编译成适合 SCI/SSCI 研究方法的学术级格式。

## 💻 API 使用方式

你也可以通过编程方式使用该框架：

```python
from swarm import Swarm, Agent
from agents import extractor_agent, analyzer_agent, reviewer_agent

client = Swarm()

# 运行完整流水线
response = client.run(
    agent=extractor_agent,
    messages=[{"role": "user", "content": "输入你的文本..."}],
    debug=True,
    max_turns=6,
)

print(response.messages[-1]["content"])
```

<details>
<summary><strong>高级 API 示例</strong></summary>

### 自定义智能体示例

```python
from swarm import Swarm, Agent

def transfer_to_expert():
    return expert_agent

triage_agent = Agent(
    name="Triage",
    instructions="将用户路由给正确的专家。",
    functions=[transfer_to_expert],
)

expert_agent = Agent(
    name="Expert",
    instructions="你是一位领域专家。请回答用户的问题。",
)

client = Swarm()
response = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "我需要关于线段树的帮助。"}],
)
```

### 结构化输出解析

流水线输出可以被解析为具有类型的 Pydantic 模型：

```python
from swarm import Swarm, Agent, AnalysisReport
from swarm.util import extract_json

client = Swarm()
response = client.run(agent=extractor_agent, messages=[...])
raw = response.messages[-1]["content"]

# 从可能被 Markdown 包裹的输出中解析 JSON
data = extract_json(raw)
if data:
    report = AnalysisReport(**data)
    print(f"问题总数: {report.total_issues}")
    print(f"严重问题: {report.critical_count}")
    for issue in report.issues:
        print(f"  [{issue.severity}] {issue.description}")
```

</details>

## 📁 项目结构

<details>
<summary><strong>查看目录树</strong></summary>

```text
Swarm-Deep-Analyzer/
├── main.py                  # 包含 CLI 参数解析的入口文件
├── agents.py                # 智能体定义（Extractor, Analyzer, Reviewer）
├── prompts/                 # 每个智能体的系统提示词
├── pyproject.toml           # 构建配置与依赖
├── LICENSE                  # MIT 许可证
├── .gitignore               # Git 忽略规则
├── .pre-commit-config.yaml  # Pre-commit 钩子配置
├── swarm/                   # 核心 Swarm 框架
│   ├── core.py              # Swarm 类（运行、流式处理、工具处理）
│   ├── types.py             # 智能体、响应、结果等类型定义
│   └── ...
├── tests/                   # 测试套件
└── examples/                # 示例应用
```

</details>

## 🧪 运行测试

```bash
# 运行所有测试
pytest

# 运行并显示详细输出
pytest -v

# 运行并附带覆盖率（需要安装 pytest-cov）
pytest --cov=swarm --cov=agents --cov=main
```

## 🤝 贡献

欢迎任何形式的贡献、问题反馈和特性请求！请随时查看 [issues 页面](https://github.com/hnuDeng/Swarm-Deep-Analyzer/issues)。
如果你觉得这个项目对你有帮助，请给它点个 ⭐️！

## 📄 许可证

MIT License. 详见 [LICENSE](LICENSE) 文件。

---
<div align="center">
Made with ❤️ by <a href="https://github.com/hnuDeng">Deng</a>
</div>
