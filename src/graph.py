"""
LangGraph 主图定义。
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.state import ResearchState
from src.agents.planner import planner_node
from src.agents.retrieval import retrieval_node
from src.agents.reading import reading_node
from src.agents.comparison import comparison_node
from src.agents.reporter import reporter_node
from src.memory.summary_memory import get_summary_memory, should_compress

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
    workflow.add_node("memory", memory_node)
    workflow.add_node("reporter", reporter_node)
    
    # === 定义边(数据流)===
    # START 是 LangGraph 的特殊起点
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retrieval")
    workflow.add_edge("retrieval", "reading")
    workflow.add_edge("reading", "comparison")
    workflow.add_edge("comparison", "memory")
    workflow.add_edge("memory", "reporter")
    workflow.add_edge("reporter", END)
    
    # 编译图
    app = workflow.compile()
    logger.info("Graph compiled successfully")
    return app

def memory_node(state: ResearchState) -> dict:
    """
    Memory 节点:条件触发 paper_summaries 压缩。
    
    触发条件(should_compress 判断):
    - 首次:paper_summaries 数 >= 10
    - 增量:新增 >= 5
    
    无论是否触发,都返回 dict(LangGraph 要求节点必须 return)。
    """
    if not should_compress(state):
        logger.info(
            f"[Memory] Skip compression "
            f"({len(state.get('paper_summaries', {}))} papers, threshold not met)"
        )
        return {}
    
    paper_summaries = state["paper_summaries"]
    logger.info(f"[Memory] Triggering compression for {len(paper_summaries)} papers")
    
    memory = get_summary_memory()
    
    # 判断是首次压缩还是增量合并
    existing = state.get("summary_memory")
    if existing is None or existing.get("covered_count", 0) == 0:
        result = memory.compress(paper_summaries)
    else:
        # 增量合并:只压缩新增的 paper
        already_covered_ids = set()  # 注意:实际工程里需要追踪哪些已被压缩
        new_papers = {
            pid: s for pid, s in paper_summaries.items()
            if pid not in already_covered_ids
        }
        result = memory.merge(existing, new_papers)
    
    logger.info(
        f"[Memory] ✅ Compressed {result['covered_count']} papers to "
        f"{len(result['summary'])} chars (method={result['compression_method']})"
    )
    
    return {"summary_memory": result}

if __name__ == "__main__":
    # 单独测试 graph 构建
    logging.basicConfig(level=logging.INFO)
    app = build_graph()
    print("✅ Graph built successfully")
    print(f"Nodes: {list(app.get_graph().nodes.keys())}")