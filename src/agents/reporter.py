"""
Reporter Agent:综合所有信息生成最终报告。
"""

import logging
from src.state import ResearchState
from src.llm import get_llm
from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

REPORTER_SYSTEM_PROMPT = """你是学术综述写作专家。根据提供的信息生成一份高质量的文献综述报告。

报告结构:
1. ## 研究背景与问题(2-3 句)
2. ## 主要研究方法综述(对比表已提供,补充叙述)
3. ## 核心发现与共识
4. ## 研究空白与未来方向(直接引用已识别的 Research Gaps)
5. ## 结论

要求:
- 中文输出,学术风格
- 每个陈述要有依据(基于提供的论文摘要)
- 研究空白部分要具体、有深度
- 全文 500-800 字
"""

REPORTER_USER_PROMPT_TEMPLATE = """研究问题:{query}

已检索论文数:{n_papers}

方法对比与 Research Gaps 分析:
{comparison_table}

相关论文背景(RAG 检索):
{rag_context}

请生成完整的文献综述报告。"""


def reporter_node(state: ResearchState) -> dict:
    logger.info("[Reporter] Generating final report")

    summary_memory = state.get("summary_memory")
    if summary_memory and summary_memory.get("summary"):
        # 用压缩 summary 替代完整 paper_summaries 描述
        compressed_context = (
            f"\n## 论文集合的压缩概览(覆盖 {summary_memory['covered_count']} 篇)\n"
            f"{summary_memory['summary']}\n"
        )
        logger.info(
            f"[Reporter] Using compressed summary "
            f"(method={summary_memory['compression_method']}, "
            f"{len(summary_memory['summary'])} chars vs "
            f"~{len(state.get('paper_summaries', {})) * 500} chars raw)"
        )
    else:
        compressed_context = ""
        logger.info("[Reporter] No summary memory, using raw paper_summaries")
    
    # === Step 1: === RAG 检索:chunk 级检索(对比旧版的"整篇匹配")===
    # 升级:从 search() 改为 search_chunks(),提升检索精度
    # 报告生成需要"多角度信息",所以 n_results 调到 5(原来 3 个整篇,现在 5 个段落)
    rag_context = ""
    try:
        vs = get_vector_store()
        if vs.collection.count() > 0:
            search_query = state.get("search_query_en") or state["query"]
            
            # ⭐ 用 search_chunks 而非 search:返回段落级匹配
            hits = vs.search_chunks_with_rerank(
                search_query,
                n_results=5,
                rerank_top_n=15,  # 召回 15,精排到 5
                enable_rerank=True,
            )
            
            if hits:
                # 在 context 中标注来源,便于 LLM 引用
                rag_context = "\n\n".join(
                    f"[来源:{h['title'][:50]} | section={h['section']} | score={h['score']:.2f}]\n"
                    f"{h['text'][:400]}"
                    for h in hits
                )
                logger.info(
                    f"[Reporter] RAG retrieved {len(hits)} chunks "
                    f"(sections: {[h['section'] for h in hits]})"
                )
    except Exception as e:
        logger.warning(f"[Reporter] RAG failed (non-critical): {e}")
        rag_context = "RAG 检索不可用"

    # === 调 LLM 生成报告 ===
    llm = get_llm()
    user_prompt = REPORTER_USER_PROMPT_TEMPLATE.format(
        query=state["query"],
        n_papers=len(state["papers"]),
        comparison_table=state["method_comparison_table"],
        rag_context=rag_context or "无额外上下文",
    ) + compressed_context 

    try:
        response = llm.chat(
            prompt=user_prompt,
            system=REPORTER_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=8192,
        )

        # 保存报告
        from pathlib import Path
        output_path = Path("output") / "final_report.md"
        output_path.write_text(response, encoding="utf-8")
        logger.info(f"[Reporter] Report saved: {output_path}")

        return {
            "final_report": response,
            "messages": [f"[Reporter] Report generated ({len(response)} chars)"],
            "step_count": state["step_count"] + 1,
        }

    except Exception as e:
        logger.error(f"[Reporter] Failed: {e}")
        fallback = f"报告生成失败。已处理 {len(state['papers'])} 篇论文。错误: {e}"
        return {
            "final_report": fallback,
            "messages": [f"[Reporter] FAILED: {e}"],
            "step_count": state["step_count"] + 1,
        }