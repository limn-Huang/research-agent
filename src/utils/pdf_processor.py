"""
PDF 处理工具:下载论文 PDF + 提取文本。

设计要点:
1. 本地缓存:同一篇论文不重复下载(用 paper_id 当文件名)
2. 失败容错:下载失败、解析失败都不让 Agent 崩溃
3. 文本切块:返回按 section 切分的结构,便于后续 LLM 处理
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict
import requests
import fitz  # PyMuPDF 的导入名是 fitz(历史原因)
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# 缓存目录(Day 1 已在 .env 配置过 DATA_DIR)
PDF_CACHE_DIR = Path("data/pdfs")
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def download_pdf(pdf_url: str, paper_id: str) -> Optional[Path]:
    """
    下载 PDF 到本地缓存。
    
    Args:
        pdf_url: PDF 下载地址
        paper_id: 论文 ID(用作文件名)
    
    Returns:
        本地 PDF 文件路径;失败返回 None
    """
    # 用 paper_id 当文件名,避免特殊字符
    safe_name = re.sub(r"[^\w\-]", "_", paper_id)
    pdf_path = PDF_CACHE_DIR / f"{safe_name}.pdf"
    
    # 缓存命中
    if pdf_path.exists() and pdf_path.stat().st_size > 1024:  # >1KB 才认为是有效文件
        logger.info(f"[PDF] Cache hit: {pdf_path}")
        return pdf_path
    
    logger.info(f"[PDF] Downloading: {pdf_url}")
    try:
        response = requests.get(
            pdf_url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (Research Agent)"},
            stream=True,  # 大文件流式下载
        )
        response.raise_for_status()  # HTTP 错误抛异常
        
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"[PDF] Downloaded: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
        return pdf_path
    
    except Exception as e:
        logger.error(f"[PDF] Download failed: {e}")
        if pdf_path.exists():
            pdf_path.unlink()  # 删除不完整文件
        return None


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    从 PDF 提取纯文本。
    
    用 PyMuPDF(fitz),它比 pypdf 快 5-10 倍,中英文支持也更好。
    
    Returns:
        提取出的纯文本;失败返回 None
    """
    if not pdf_path or not pdf_path.exists():
        return None
    
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            text_parts.append(text)
        doc.close()
        
        full_text = "\n".join(text_parts)
        logger.info(f"[PDF] Extracted {len(full_text)} chars from {len(text_parts)} pages")
        return full_text
    
    except Exception as e:
        logger.error(f"[PDF] Extract failed: {e}")
        return None


def truncate_for_llm(text: str, max_chars: int = 12000) -> str:
    """
    截断长文本以适配 LLM context。
    
    策略:取开头 + 结尾(论文的关键信息通常在 abstract、intro 和 conclusion)
    """
    if len(text) <= max_chars:
        return text
    
    head_size = int(max_chars * 0.6)  # 60% 给开头(含 abstract、intro)
    tail_size = max_chars - head_size  # 40% 给结尾(含 conclusion)
    
    head = text[:head_size]
    tail = text[-tail_size:]
    
    return f"{head}\n\n... [中间内容已省略] ...\n\n{tail}"


def fetch_paper_text(pdf_url: str, paper_id: str) -> Optional[str]:
    """
    一站式接口:给 URL 和 ID,返回可送进 LLM 的文本。
    
    流程:下载 → 解析 → 截断
    """
    pdf_path = download_pdf(pdf_url, paper_id)
    if not pdf_path:
        return None
    
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text:
        return None
    
    truncated = truncate_for_llm(full_text)
    return truncated


if __name__ == "__main__":
    # 单测:用一个真实 arXiv PDF 测试
    logging.basicConfig(level=logging.INFO)
    
    test_url = "https://arxiv.org/pdf/2401.00001v1.pdf"  # 任意一个 arXiv PDF
    test_id = "test_paper_001"
    
    text = fetch_paper_text(test_url, test_id)
    if text:
        print(f"\n✅ Got text, length: {len(text)} chars")
        print(f"\n--- First 500 chars ---")
        print(text[:500])
        print(f"\n--- Last 500 chars ---")
        print(text[-500:])
    else:
        print("❌ Failed to fetch paper")