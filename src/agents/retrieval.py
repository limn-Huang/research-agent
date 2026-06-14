"""
Retrieval Agent - 真实数据源版本。

设计要点:
1. 通过 BaseRetriever 解耦,可以无缝切换数据源
2. 支持多源并行检索
3. 失败时返回空列表,让后续节点能 graceful degrade
"""

import logging
from src.state import ResearchState
from src.retrievers.base import get_retriever

logger = logging.getLogger(__name__)


def retrieval_node(state: ResearchState) -> dict:
    # 优先用 Planner 生成的英文检索词,没有则 fallback 用原始 query
    query = state.get("search_query_en") or state["query"]
    max_papers = state["max_papers"]

    logger.info(f"[Retrieval] English query: '{query}'")
    # === 检索逻辑 ===
    try:
        retriever = get_retriever("arxiv")
        papers = retriever.search(query=query, max_results=max_papers)
        
        if not papers:
            logger.warning("[Retrieval] No papers found")
            return {
                "papers": [],
                "messages": ["[Retrieval] No papers found"],
                "step_count": state["step_count"] + 1,
            }
        
        return {
            "papers": papers,
            "messages": [f"[Retrieval] Found {len(papers)} papers from arxiv"],
            "step_count": state["step_count"] + 1,
        }
    
    except Exception as e:
        logger.error(f"[Retrieval] Failed: {e}")
        return {
            "papers": [],
            "error": f"Retrieval failed: {str(e)}",
            "messages": [f"[Retrieval] FAILED: {e}"],
            "step_count": state["step_count"] + 1,
        }