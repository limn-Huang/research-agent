"""
SummaryMemory —— 通用上下文压缩模块。

设计目标(与 OpenHands 项目的 SummaryMemory 共享):
1. 把累积的 state 数据用 LLM 压缩成简短摘要
2. 不破坏原始数据(只输出 summary,原数据保留)
3. 支持增量合并(已有 summary + 新数据 → 新 summary)
4. 失败容错(LLM 调用失败时返回 fallback summary)

在 research-agent 中的具体场景:
- LangGraph 的 ResearchState 累积 paper_summaries(每篇 ~500 token)
- 当处理 30+ 篇论文时,state 膨胀,后续节点 prompt 爆炸
- 触发压缩:把 paper_summaries 列表压缩为简短的"研究图景概览"
- 保留:method 共性、finding 共识、关键 limitation;
  丢弃:每篇论文的元数据、重复表述、冗长描述

与 OpenHands 项目的复用关系:
- 同一种"用 LLM 压缩 + 失败 fallback + 增量 merge"模式
- 两个项目通过不同的"输入数据格式 + 触发条件"接入
- 验证了 SummaryMemory 作为**领域无关基础设施**的可行性

"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

from src.llm import get_llm

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt 模板
# =============================================================================

COMPRESS_SYSTEM_PROMPT = """你是一个学术文献摘要压缩专家。

你的任务是把多篇论文的结构化摘要,压缩成一段简短的"研究图景概览"。

压缩原则:
1. 保留各论文的方法共性与分歧(它们用了哪些不同方法?)
2. 保留主要发现的共识(它们的核心结论是什么?)
3. 保留关键 limitation(整体研究空白在哪?)
4. 丢弃:论文标题、作者、数据集名称等元信息;重复表述;冗长细节

输出格式:
- 第一段(2-3 句):研究领域总体方法图景
- 第二段(2-3 句):核心发现与共识
- 第三段(1-2 句):关键 limitation 与未来方向

总字数控制在 200-300 字之间(原始内容可能 3000+ 字,压缩比约 10x)。

直接输出压缩后的文本,不要 markdown 包裹,不要解释。
"""

COMPRESS_USER_TEMPLATE = """请压缩以下 {n_papers} 篇论文的摘要:

{papers_text}

按 system prompt 中的格式输出压缩后的"研究图景概览"。
"""

MERGE_SYSTEM_PROMPT = """你是一个学术摘要增量压缩专家。

你会收到两部分内容:
1. 已有的"研究图景概览"(之前 N 篇论文的压缩结果)
2. 新增加的若干篇论文摘要

请生成一个新的"研究图景概览",**整合两部分信息**,保持总字数 200-300 字。

输出格式同前:三段式(方法 / 发现 / limitation),直接输出文本不要 markdown 包裹。
"""

MERGE_USER_TEMPLATE = """# 已有的研究图景概览(覆盖 {prev_count} 篇论文):

{previous_summary}

# 新增 {n_new} 篇论文摘要:

{new_papers_text}

请整合输出新的"研究图景概览"。
"""


# =============================================================================
# 工具函数
# =============================================================================

def _format_papers_for_compress(paper_summaries: dict, max_chars_per_paper: int = 600) -> str:
    """
    把 paper_summaries dict 转成给 LLM 看的文本格式。
    
    paper_summaries 结构(来自 src/state.py):
        {
            "paper_id_1": {"method": "...", "dataset": "...", "finding": "...", "limitation": "..."},
            "paper_id_2": {...},
        }
    """
    if not paper_summaries:
        return "(empty)"
    
    lines = []
    for i, (paper_id, summary) in enumerate(paper_summaries.items(), 1):
        # 截断长字段(避免某篇论文主导整个 prompt)
        def truncate(s, n):
            s = str(s) if s else ""
            return s[:n] + ("..." if len(s) > n else "")
        
        text = (
            f"### 论文 {i} (id={paper_id[:30]}):\n"
            f"- 方法: {truncate(summary.get('method', ''), max_chars_per_paper // 2)}\n"
            f"- 数据: {truncate(summary.get('dataset', ''), 100)}\n"
            f"- 发现: {truncate(summary.get('finding', ''), max_chars_per_paper // 2)}\n"
            f"- 局限: {truncate(summary.get('limitation', ''), 100)}\n"
        )
        lines.append(text)
    
    return "\n".join(lines)


# =============================================================================
# 核心类:SummaryMemory
# =============================================================================

class SummaryMemory:
    """
    通用 state 压缩模块(research-agent 版本)。
    
    输入:paper_summaries dict
    输出:压缩后的 summary 文本 + 元数据 dict
    
    使用方式:
        memory = SummaryMemory()
        result = memory.compress(state["paper_summaries"])
        # result = {
        #     "summary": "...",
        #     "covered_count": 30,
        #     "version": 1,
        #     "last_compressed_at": "2026-...",
        # }
    """
    
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
    
    def compress(self, paper_summaries: dict) -> dict:
        """
        从 0 压缩:把 N 个 paper_summaries 压缩成 1 个 summary。
        
        Args:
            paper_summaries: state["paper_summaries"] 的 dict
        
        Returns:
            dict 含 summary / covered_count / version / 时间戳 / 方法
        """
        if not paper_summaries:
            return self._empty_summary()
        
        n_papers = len(paper_summaries)
        logger.info(f"[SummaryMemory] Compressing {n_papers} paper summaries...")
        
        papers_text = _format_papers_for_compress(paper_summaries)
        user_prompt = COMPRESS_USER_TEMPLATE.format(
            n_papers=n_papers,
            papers_text=papers_text,
        )
        
        try:
            llm = get_llm()
            response = llm.chat(
                prompt=user_prompt,
                system=COMPRESS_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=self.max_tokens,
            )
            
            summary_text = response.strip() if response else ""
            
            if not summary_text:
                logger.warning("[SummaryMemory] LLM returned empty")
                return self._fallback_summary(paper_summaries, reason="empty LLM response")
            
            return {
                "summary": summary_text,
                "covered_count": n_papers,
                "version": 1,
                "last_compressed_at": datetime.now(timezone.utc).isoformat(),
                "compression_method": "llm",
            }
        
        except Exception as e:
            logger.error(f"[SummaryMemory] LLM call failed: {e}", exc_info=True)
            return self._fallback_summary(paper_summaries, reason=f"LLM error: {e}")
    
    def merge(
        self,
        previous_summary: dict,
        new_paper_summaries: dict,
    ) -> dict:
        """
        增量合并:已有 summary + 新 paper_summaries → 新 summary。
        
        场景:第一次压缩 30 篇,后续又新增 10 篇,不重新压全部,只合并新的。
        """
        if not new_paper_summaries:
            return previous_summary
        
        prev_text = previous_summary.get("summary", "")
        prev_count = previous_summary.get("covered_count", 0)
        n_new = len(new_paper_summaries)
        
        new_papers_text = _format_papers_for_compress(new_paper_summaries)
        user_prompt = MERGE_USER_TEMPLATE.format(
            prev_count=prev_count,
            previous_summary=prev_text,
            n_new=n_new,
            new_papers_text=new_papers_text,
        )
        
        try:
            llm = get_llm()
            response = llm.chat(
                prompt=user_prompt,
                system=MERGE_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=self.max_tokens,
            )
            
            merged_text = response.strip() if response else ""
            
            if not merged_text:
                return self._fallback_summary(new_paper_summaries, reason="merge empty response")
            
            return {
                "summary": merged_text,
                "covered_count": prev_count + n_new,
                "version": previous_summary.get("version", 1) + 1,
                "last_compressed_at": datetime.now(timezone.utc).isoformat(),
                "compression_method": "llm_merge",
            }
        
        except Exception as e:
            logger.error(f"[SummaryMemory] Merge failed: {e}", exc_info=True)
            return self._fallback_summary(new_paper_summaries, reason=f"merge error: {e}")
    
    def _empty_summary(self) -> dict:
        return {
            "summary": "",
            "covered_count": 0,
            "version": 0,
            "last_compressed_at": datetime.now(timezone.utc).isoformat(),
            "compression_method": "empty",
        }
    
    def _fallback_summary(self, paper_summaries: dict, reason: str) -> dict:
        """
        LLM 失败时的降级:列出论文 id 数量,至少给后续 hook 一个"我知道压缩失败了"的信号。
        """
        paper_ids = list(paper_summaries.keys())[:5]
        preview = ", ".join(p[:20] for p in paper_ids)
        
        return {
            "summary": f"[Fallback: {reason}] 共 {len(paper_summaries)} 篇论文,样本: {preview}...",
            "covered_count": len(paper_summaries),
            "version": 1,
            "last_compressed_at": datetime.now(timezone.utc).isoformat(),
            "compression_method": "fallback",
            "fallback_reason": reason,
        }


# =============================================================================
# 触发判断逻辑(供 graph.py 调用)
# =============================================================================

DEFAULT_COMPRESSION_THRESHOLD = 10  # 累积到 10 篇 paper_summary 触发首次压缩
DEFAULT_INCREMENT_THRESHOLD = 5      # 之后每 5 篇增量合并


def should_compress(
    state: dict,
    initial_threshold: int = DEFAULT_COMPRESSION_THRESHOLD,
    increment_threshold: int = DEFAULT_INCREMENT_THRESHOLD,
) -> bool:
    """
    判断是否需要触发压缩。
    
    触发条件:
    1. paper_summaries 数 >= initial_threshold,且还没压缩过(首次压缩)
    2. paper_summaries 数 - 已压缩数 >= increment_threshold(增量压缩)
    """
    paper_summaries = state.get("paper_summaries", {})
    n_papers = len(paper_summaries)
    
    summary_state = state.get("summary_memory")
    
    # Case 1: 首次压缩
    if summary_state is None:
        return n_papers >= initial_threshold
    
    # Case 2: 增量
    already_covered = summary_state.get("covered_count", 0)
    new_count = n_papers - already_covered
    return new_count >= increment_threshold


# =============================================================================
# 全局单例
# =============================================================================

_global_memory: Optional[SummaryMemory] = None


def get_summary_memory() -> SummaryMemory:
    """获取全局 SummaryMemory 单例"""
    global _global_memory
    if _global_memory is None:
        _global_memory = SummaryMemory()
    return _global_memory


# =============================================================================
# 单测(python -m src.memory.summary_memory)
# =============================================================================

if __name__ == "__main__":
    import logging
    from dotenv import load_dotenv
    
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # 模拟一个 state
    fake_paper_summaries = {
        f"paper_{i}": {
            "method": f"Method {i}: We propose technique X with attention to detail.",
            "dataset": f"Dataset_{i}",
            "finding": f"Finding {i}: Significantly outperforms baseline by 10-20%.",
            "limitation": f"Limitation {i}: Requires large-scale data.",
        }
        for i in range(1, 6)  # 5 篇模拟论文
    }
    
    print("=" * 70)
    print("Test 1: 实例化 + should_compress")
    print("=" * 70)
    
    memory = get_summary_memory()
    state_with_few = {"paper_summaries": fake_paper_summaries}
    state_with_many = {"paper_summaries": {**fake_paper_summaries, **{f"paper_{i}": fake_paper_summaries["paper_1"] for i in range(6, 12)}}}
    
    print(f"5 papers, should compress: {should_compress(state_with_few)}")  # False
    print(f"11 papers, should compress: {should_compress(state_with_many)}")  # True
    
    print("\n" + "=" * 70)
    print("Test 2: Compress (调真实 LLM)")
    print("=" * 70)
    
    result = memory.compress(fake_paper_summaries)
    print(f"\n压缩结果:")
    print(f"  Method: {result['compression_method']}")
    print(f"  Covered: {result['covered_count']}")
    print(f"  Version: {result['version']}")
    print(f"  Summary length: {len(result['summary'])} chars")
    print(f"\n--- Summary 内容 ---")
    print(result['summary'])
    
    print("\n" + "=" * 70)
    print("Test 3: 空压缩 + Fallback")
    print("=" * 70)
    
    empty = memory.compress({})
    print(f"Empty: {empty['compression_method']} (should be 'empty')")
    
    fallback = memory._fallback_summary(fake_paper_summaries, reason="test")
    print(f"Fallback: {fallback['summary'][:100]}...")
    
    print("\n✅ All summary memory tests done")