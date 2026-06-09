"""
Comparison Agent 真实版:用 LLM 生成结构化方法对比表 + 识别 Research Gap。

设计要点:
1. 把所有 paper_summaries 整理成结构化输入给 LLM
2. 要求 LLM 输出 markdown 表格 + research gap 分析
3. 同时把论文存入 VectorStore(为 Reporter 的 RAG 检索做准备)
"""

import logging
from src.state import ResearchState
from src.llm import get_llm
from src.rag.vector_store import get_vector_store


logger = logging.getLogger(__name__)

COMPARISON_SYSTEM_PROMPT = """你是学术文献分析专家,擅长对比不同研究的方法论和发现。

输出要求:
1. 先输出一个 markdown 表格,列为:论文标题、研究方法、数据集/规模、核心发现、局限性
2. 表格后输出"## Research Gaps"章节,列出 3-5 个已识别的研究空白
3. Research Gap 要具体,如"现有研究缺少 X 维度的考察"而非"还需要更多研究"
4. 全部用中文输出
"""

COMPARISON_USER_PROMPT_TEMPLATE = """研究主题:{query}

以下是 {n_papers} 篇相关论文的结构化摘要:

{papers_text}

请生成:
1. 方法对比表(markdown 格式)
2. Research Gaps 分析(## Research Gaps 标题下,列出 3-5 条)
"""


def build_papers_text(paper_summaries: dict, papers: list) -> str:
    """把 paper_summaries 整理成 LLM 友好的文本格式"""
    lines = []
    paper_map = {p["paper_id"]: p for p in papers}

    for i, (paper_id, summary) in enumerate(paper_summaries.items(), 1):
        paper = paper_map.get(paper_id, {})
        title = paper.get("title", paper_id)[:80]
        lines.append(f"""
**论文 {i}: {title}**
- 方法: {summary['method']}
- 数据: {summary['dataset']}
- 发现: {summary['finding']}
- 局限: {summary['limitation']}
""")
    return "\n".join(lines)


def comparison_node(state: ResearchState) -> dict:
    """Comparison 节点:生成对比表 + 把论文存入 VectorStore"""
    paper_summaries = state["paper_summaries"]
    papers = state["papers"]

    logger.info(f"[Comparison] Comparing {len(paper_summaries)} papers")

    if not paper_summaries:
        return {
            "method_comparison_table": "No papers to compare.",
            "messages": ["[Comparison] No summaries available"],
            "step_count": state["step_count"] + 1,
        }

    # === Step 1: 把论文存入 VectorStore ===
    try:
        vs = get_vector_store()
        added = vs.add_papers(papers, paper_summaries)
        logger.info(f"[Comparison] Added {added} papers to VectorStore")
    except Exception as e:
        logger.warning(f"[Comparison] VectorStore failed (non-critical): {e}")

    # === Step 2: 调 LLM 生成对比 ===
    llm = get_llm()
    papers_text = build_papers_text(paper_summaries, papers)
    user_prompt = COMPARISON_USER_PROMPT_TEMPLATE.format(
        query=state["query"],
        n_papers=len(paper_summaries),
        papers_text=papers_text,
    )

    try:
        response = llm.chat(
            prompt=user_prompt,
            system=COMPARISON_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=8192,
        )
        return {
            "method_comparison_table": response,
            "messages": [f"[Comparison] Generated comparison for {len(paper_summaries)} papers"],
            "step_count": state["step_count"] + 1,
        }

    except Exception as e:
        logger.error(f"[Comparison] LLM failed: {e}")
        # Fallback:至少给一个有数据的表格
        fallback_table = "| 论文 | 方法 | 数据 | 发现 | 局限 |\n|---|---|---|---|---|\n"
        for pid, s in paper_summaries.items():
            fallback_table += f"| {pid[:20]} | {s['method'][:30]} | {s['dataset'][:20]} | {s['finding'][:30]} | {s['limitation'][:20]} |\n"
        return {
            "method_comparison_table": fallback_table,
            "messages": [f"[Comparison] LLM failed, used fallback table"],
            "step_count": state["step_count"] + 1,
        }