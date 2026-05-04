import os
from swarm import Swarm
from agents import extractor_agent

# 初始化 Swarm 客户端
client = Swarm()

def run_deep_analysis(massive_text):
    print("🚀 [System] 启动基于 Swarm 的多 Agent 深度分析流...")
    print("⚠️ [Warning] 该过程涉及无状态上下文交接，将产生极大的 Token 吞吐量。")
    
    # 核心执行流：Swarm 会自动处理 Agent 之间的路由和超长上下文的合并
    response = client.run(
        agent=extractor_agent,
        messages=[{"role": "user", "content": f"请对以下超长材料进行深度结构化分析:\n\n{massive_text}"}],
        debug=True # 开启 Debug 模式，用于在控制台观察庞大的 Token 交接日志
    )
    
    print("\n✅ [System] 分析完成。最终报告：")
    print(response.messages[-1]["content"])

if __name__ == "__main__":
    # 模拟输入一个超长的业务逻辑或论文文本
    mock_long_context = " [此处为预留的超大文本输入接口，例如 50,000 字的系统架构文档或学术文献] " * 50
    run_deep_analysis(mock_long_context)