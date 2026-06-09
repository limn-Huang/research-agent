"""
LangGraph 主图定义。

这是整个项目的"大脑"——决定 5 个 agent 节点怎么连接、怎么流转。

当前架构(Day 1 简化版,线性流):
    START → Planner → Retrieval → Reading → Comparison → Reporter → END

后续(Day 4 完整版):
    引入 conditional edge,根据状态决定是否需要补充检索等。
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.state import ResearchState
from src.agents.planner import planner_node
from src.agents.retrieval import retrieval_node
from src.agents.reading import reading_node
from src.agents.comparison import comparison_node
from src.agents.reporter import reporter_node


logger = logging.getLogger(__name__)


def build_graph():
    """
    构建 ResearchAgent 的 StateGraph。
    
    StateGraph 工作流程:
    1. 定义节点:每个节点是一个函数,接收 state 返回 dict
    2. 定义边:决定执行顺序
    3. compile():把图编译成可调用对象
    
    返回:
        编译后的 graph,可以 .invoke(state) 调用
    """
    
    # 用 ResearchState 作为状态类型(LangGraph 会用它做类型检查)
    workflow = StateGraph(ResearchState)
    
    # === 注册 5 个节点 ===
    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("reading", reading_node)
    workflow.add_node("comparison", comparison_node)
    workflow.add_node("reporter", reporter_node)
    
    # === 定义边(数据流)===
    # START 是 LangGraph 的特殊起点
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retrieval")
    workflow.add_edge("retrieval", "reading")
    workflow.add_edge("reading", "comparison")
    workflow.add_edge("comparison", "reporter")
    workflow.add_edge("reporter", END)
    
    # 编译图
    app = workflow.compile()
    logger.info("Graph compiled successfully")
    return app


if __name__ == "__main__":
    # 单独测试 graph 构建
    logging.basicConfig(level=logging.INFO)
    app = build_graph()
    print("✅ Graph built successfully")
    print(f"Nodes: {list(app.get_graph().nodes.keys())}")