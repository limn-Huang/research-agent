"""
arXiv 数据源实现。

使用 arxiv 库(已在 requirements.txt 里),它是 arXiv API 的官方 Python 客户端。
特点:
- 免费、稳定、不需要 API key
- 返回的论文有完整元数据(标题、作者、摘要、PDF 链接、年份)
- 支持中英文 query(虽然主要是英文论文)
"""

import logging
import arxiv
from typing import List
from datetime import datetime

from src.retrievers.base import BaseRetriever
from src.state import Paper

logger = logging.getLogger(__name__)


class ArxivRetriever(BaseRetriever):
    """arXiv 论文检索器"""
    
    @property
    def name(self) -> str:
        return "arxiv"
    
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """
        在 arXiv 上搜索论文。
        
        Args:
            query: 搜索关键词(建议英文,arXiv 主要是英文论文)
            max_results: 最多返回多少篇
        
        Returns:
            Paper 列表,按相关性排序
        """
        logger.info(f"[ArxivRetriever] Searching: '{query}', max={max_results}")
        
        try:
            # arxiv.Search 是这个库的核心 API
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,  # 按相关性排序
                sort_order=arxiv.SortOrder.Descending,
            )
            
            # arxiv.Client 是 0.6+ 版本推荐的调用方式(旧版直接 search.results())
            client = arxiv.Client(
                page_size=max_results,
                delay_seconds=3,    # arXiv 限频:每次请求间隔 3 秒
                num_retries=3,      # 失败重试
            )
            
            papers = []
            for result in client.results(search):
                paper = self._convert_to_paper(result)
                papers.append(paper)
            
            logger.info(f"[ArxivRetriever] Got {len(papers)} papers")
            return papers
        
        except Exception as e:
            logger.error(f"[ArxivRetriever] Search failed: {e}")
            return []  # 失败返空列表,而不是 raise(让 Agent 能继续走 fallback)
    
    def _convert_to_paper(self, arxiv_result) -> Paper:
        """把 arxiv 库的 Result 对象转换为我们的 Paper 格式"""
        # arxiv ID 形如 "http://arxiv.org/abs/2401.12345v1",我们只取 ID 部分
        arxiv_id = arxiv_result.entry_id.split("/")[-1]
        
        return Paper(
            paper_id=f"arxiv_{arxiv_id}",
            title=arxiv_result.title.strip(),
            authors=[author.name for author in arxiv_result.authors],
            abstract=arxiv_result.summary.strip(),
            year=arxiv_result.published.year if arxiv_result.published else 0,
            pdf_url=arxiv_result.pdf_url,
            source="arxiv",
        )


if __name__ == "__main__":
    # 单测:python -m src.retrievers.arxiv_retriever
    logging.basicConfig(level=logging.INFO)
    
    retriever = ArxivRetriever()
    papers = retriever.search("housing vacancy carbon emission", max_results=3)
    
    print(f"\n✅ Found {len(papers)} papers:\n")
    for i, p in enumerate(papers, 1):
        print(f"{i}. {p['title']}")
        print(f"   Year: {p['year']}, Authors: {', '.join(p['authors'][:2])}")
        print(f"   PDF: {p['pdf_url']}")
        print(f"   Abstract: {p['abstract'][:150]}...\n")