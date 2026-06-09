"""
向量存储模块:把论文摘要 embedding 后存入 ChromaDB。

为什么要 RAG?
当前 Reading Agent 把整篇论文喂给 LLM,但:
1. 论文太长会超出 context window
2. 不是所有内容都相关,注入无关内容会降低质量
3. 未来论文库变大,不可能每次都全量处理

RAG 解决方案:
- 离线:把论文内容切块 embedding 存入向量库
- 在线:用户 query embedding 后检索最相关的 chunk
- 只把相关 chunk 送给 LLM

今天我们先做"按 paper 粒度"的向量化(每篇论文的摘要+提取摘要作为一个向量)
Day 5 做"按 chunk 粒度"的细粒度 RAG
"""

import logging
import os
from typing import List, Optional
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.state import Paper, PaperSummary

logger = logging.getLogger(__name__)

# ChromaDB 本地持久化路径
CHROMA_DIR = Path("data/chroma_db")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Collection 名字(类似数据库里的"表名")
COLLECTION_NAME = "research_papers"


def get_embedding_function():
    """
    返回 embedding 函数。

    这里用 GLM 的 embedding 模型。
    ChromaDB 支持自定义 embedding function,我们包装一下 GLM。
    """
    api_key = os.getenv("ZHIPUAI_API_KEY")
    base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    model = os.getenv("GLM_EMBEDDING_MODEL", "embedding-3")

    # 用 OpenAI 兼容接口包装 GLM embedding
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=model,
        api_base=base_url,
    )


class PaperVectorStore:
    """
    论文向量存储。

    封装 ChromaDB 的增删改查,提供面向业务的接口。

    设计决策:
    - 用持久化 ChromaDB(不是内存版),跨 session 保留数据
    - 每篇论文以 paper_id 为主键(避免重复入库)
    - 存储内容:摘要 + 结构化提取(method/finding)拼接后 embedding
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.ef = get_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},  # 用 cosine 相似度
        )
        logger.info(
            f"[VectorStore] Collection '{COLLECTION_NAME}' "
            f"has {self.collection.count()} documents"
        )

    def add_papers(
        self,
        papers: List[Paper],
        summaries: dict,  # paper_id -> PaperSummary
    ) -> int:
        """
        把论文信息向量化并存入 ChromaDB。

        存储的文本 = abstract + 结构化摘要(method + finding)
        这样检索时能同时匹配"原始描述"和"提取的关键信息"

        Returns:
            成功入库的论文数
        """
        ids = []
        documents = []
        metadatas = []

        for paper in papers:
            paper_id = paper["paper_id"]

            # 避免重复入库
            existing = self.collection.get(ids=[paper_id])
            if existing["ids"]:
                logger.debug(f"[VectorStore] Skip (already exists): {paper_id}")
                continue

            # 拼接要 embedding 的文本
            summary = summaries.get(paper_id)
            if summary:
                doc_text = (
                    f"Title: {paper['title']}\n"
                    f"Abstract: {paper['abstract']}\n"
                    f"Method: {summary['method']}\n"
                    f"Finding: {summary['finding']}"
                )
            else:
                doc_text = f"Title: {paper['title']}\n Abstract: {paper['abstract']}"

            ids.append(paper_id)
            documents.append(doc_text)
            metadatas.append(
                {
                    "title": paper["title"][:500],
                    "year": paper["year"],
                    "source": paper["source"],
                    "authors": ", ".join(paper["authors"][:3]),
                }
            )

        if not ids:
            logger.info("[VectorStore] Nothing new to add")
            return 0

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(f"[VectorStore] Added {len(ids)} papers")
        return len(ids)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> List[dict]:
        """
        语义检索:找和 query 最相关的论文。

        Args:
            query: 检索问题
            n_results: 返回几篇
            where: metadata 过滤条件,如 {"year": {"$gte": 2020}}

        Returns:
            [{"paper_id", "title", "score", "document"}, ...]
        """
        count = self.collection.count()
        if count == 0:
            logger.warning("[VectorStore] Collection is empty")
            return []

        n_results = min(n_results, count)

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i, paper_id in enumerate(results["ids"][0]):
            hits.append(
                {
                    "paper_id": paper_id,
                    "title": results["metadatas"][0][i]["title"],
                    "score": 1 - results["distances"][0][i],  # cosine: 距离越小越相关
                    "document": results["documents"][0][i],
                }
            )

        logger.info(f"[VectorStore] Query '{query[:50]}' → {len(hits)} hits")
        return hits


# 全局单例
_vector_store: Optional[PaperVectorStore] = None


def get_vector_store() -> PaperVectorStore:
    """获取全局 VectorStore 单例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = PaperVectorStore()
    return _vector_store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()

    vs = get_vector_store()
    print(f"Collection count: {vs.collection.count()}")
    print("✅ VectorStore OK")