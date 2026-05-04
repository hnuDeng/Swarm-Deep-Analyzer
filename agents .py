from swarm import Agent

# 定义交接函数 (Handoff Functions)
def transfer_to_analyzer():
    """将提取好的超长上下文和初始分析移交给深度逻辑分析师。"""
    return analyzer_agent

def transfer_to_reviewer():
    """将分析报告和原始上下文移交给最终审查员。"""
    return reviewer_agent

# 1. 提取者 Agent (负责读取巨量文本)
extractor_agent = Agent(
    name="Context Extractor",
    instructions="""你是一个顶级的上下文提取专家。
    你的任务是读取用户输入的超长文本（如数十页的学术文献或庞大的 C++/Python 代码库）。
    请提取其中的核心逻辑架构、算法复杂度瓶颈或核心论点。
    完成后，必须调用 transfer_to_analyzer 将全部上下文移交。""",
    functions=[transfer_to_analyzer],
)

# 2. 分析师 Agent (负责消耗 Token 进行长链推理)
analyzer_agent = Agent(
    name="Deep Logic Analyzer",
    instructions="""你是一个深度的逻辑分析师。你接收了提取者的全部上下文历史。
    请运用多步思维链（Chain-of-Thought），对文本中的逻辑缺陷、底层数据结构问题或情感矛盾进行推演。
    由于上下文极长，请务必保持逻辑严密。推理完成后，调用 transfer_to_reviewer。""",
    functions=[transfer_to_reviewer],
)

# 3. 审查员 Agent (最终输出)
reviewer_agent = Agent(
    name="Final Reviewer",
    instructions="""你是最终审查员。仔细阅读前面所有 Agent 的交互历史。
    输出一份结构化的 JSON 格式报告，指出具体问题点和修复建议。""",
)