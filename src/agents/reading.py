"""
Reading Agent - 用 LLM 从论文中提取结构化信息。

设计要点:
1. Prompt 严格要求 JSON 输出(便于程序化处理)
2. 处理 LLM 输出的常见问题(markdown 包裹、字段缺失)
3. 并发处理多篇论文(asyncio 在 Day 5 引入,先用串行)
4. 失败时使用 abstract fallback,不让单篇失败拖垮整体
"""

import json
import logging
import re
from typing import Optional

from src.state import ResearchState, PaperSummary, Paper
from src.llm import get_llm
from src.utils.pdf_processor import fetch_paper_text

logger = logging.getLogger(__name__)


READING_SYSTEM_PROMPT = """你是学术论文分析专家。你的任务是从论文文本中提取结构化的核心信息。

输出要求:
1. 必须输出合法的 JSON,不要任何额外文字、markdown 标记、解释
2. 包含 4 个字段:method, dataset, finding, limitation
3. 每个字段限 1-3 句话,精炼准确
4. 用客观的第三方视角描述,不要用"作者声称"这种引述
5. 如果论文中没有明确信息,该字段写 "Not specified"
"""


READING_USER_PROMPT_TEMPLATE = """以下是一篇论文的内容(可能已被截断,保留了头部和尾部):

标题: {title}
作者: {authors}

正文:
{text}

请提取以下 4 项信息,输出 JSON 格式:
{{
  "method": "用了什么方法/技术/模型(1-3句)",
  "dataset": "用了什么数据/数据集/实验对象(1-3句)",
  "finding": "主要发现/结论是什么(1-3句)",
  "limitation": "论文的局限性是什么(1-3句)"
}}

只输出 JSON,不要 markdown 代码块,不要解释。"""


def extract_json(text: str) -> str:
    """从 LLM 输出中提取 JSON(去掉 markdown 包裹等)"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_summary(llm_output: str, paper_id: str) -> Optional[PaperSummary]:
    """解析 LLM 输出为 PaperSummary;失败返回 None"""
    # ⭐ 关键改进:先检查是否空输出
    if not llm_output or not llm_output.strip():
        logger.error(f"[Reading] LLM returned EMPTY output for {paper_id}")
        return None
    
    try:
        cleaned = extract_json(llm_output)
        if not cleaned:
            logger.error(f"[Reading] After cleaning, output is empty for {paper_id}")
            return None
        
        data = json.loads(cleaned)
        
        return PaperSummary(
            paper_id=paper_id,
            method=data.get("method", "Not specified"),
            dataset=data.get("dataset", "Not specified"),
            finding=data.get("finding", "Not specified"),
            limitation=data.get("limitation", "Not specified"),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"[Reading] Parse failed for {paper_id}: {e}")
        logger.warning(f"[Reading] Raw output FULL (first 1000 chars): {llm_output[:1000]!r}")  # ⭐ 用 !r 显示真实字符(含换行)
        return None


def summarize_paper(paper: Paper) -> PaperSummary:
    paper_id = paper["paper_id"]
    logger.info(f"[Reading] Summarizing: {paper['title'][:60]}...")
    
    text = None
    if paper["pdf_url"]:
        text = fetch_paper_text(paper["pdf_url"], paper_id)
    
    if not text:
        logger.warning(f"[Reading] PDF unavailable, using abstract for {paper_id}")
        text = paper["abstract"]
    
    # ⭐ 新增:让 PDF 文本更短一些(8000 字符,留足够空间给 prompt 模板和输出)
    if len(text) > 8000:
        text = text[:5000] + "\n\n... [中间省略] ...\n\n" + text[-3000:]
    
    logger.info(f"[Reading] Text length sent to LLM: {len(text)} chars")  # ⭐ 关键诊断日志
    
    llm = get_llm()
    user_prompt = READING_USER_PROMPT_TEMPLATE.format(
        title=paper["title"],
        authors=", ".join(paper["authors"][:3]),
        text=text,
    )
    
    try:
        response = llm.chat(
            prompt=user_prompt,
            system=READING_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=4096,
        )
        # ⭐ 新增:不管解析成功失败,先打印 raw output 长度
        logger.info(f"[Reading] LLM raw response length: {len(response)} chars")
        if not response or not response.strip():
            logger.error(f"[Reading] LLM returned EMPTY for {paper_id}")
        else:
            logger.info(f"[Reading] LLM response preview: {response[:200]!r}")
        
        summary = parse_summary(response, paper_id)
        if summary:
            return summary
    except Exception as e:
        logger.error(f"[Reading] LLM failed for {paper_id}: {e}")
    
    return PaperSummary(
        paper_id=paper_id,
        method="Extraction failed",
        dataset="Extraction failed",
        finding=paper["abstract"][:200] + "...",
        limitation="Not analyzed",
    )


def reading_node(state: ResearchState) -> dict:
    """
    Reading 节点:逐篇处理所有 papers,生成 PaperSummary。
    
    当前是串行处理 —— Day 5+ 可改为 asyncio 并发,提升 3-5 倍速度。
    """
    papers = state["papers"]
    logger.info(f"[Reading] Processing {len(papers)} papers")
    
    if not papers:
        return {
            "paper_summaries": {},
            "messages": ["[Reading] No papers to process"],
            "step_count": state["step_count"] + 1,
        }
    
    summaries = {}
    success_count = 0
    
    for i, paper in enumerate(papers, 1):
        logger.info(f"[Reading] Paper {i}/{len(papers)}")
        try:
            summary = summarize_paper(paper)
            summaries[paper["paper_id"]] = summary
            if summary["method"] != "Extraction failed":
                success_count += 1
        except Exception as e:
            logger.error(f"[Reading] Unexpected error for {paper['paper_id']}: {e}")
            continue
    
    return {
        "paper_summaries": summaries,
        "messages": [f"[Reading] Summarized {success_count}/{len(papers)} papers successfully"],
        "step_count": state["step_count"] + 1,
    }