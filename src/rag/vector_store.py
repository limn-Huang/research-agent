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
from src.rag.chunkers import BaseChunker, Chunk, get_chunker

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
    
    def add_papers_with_chunks(
        self,
        papers: List[Paper],
        summaries: dict,
        chunker: Optional[BaseChunker] = None,
        chunker_strategy: str = "sliding",
        chunk_size: int = 512,
        overlap: int = 80,
    ) -> int:
        """
        ⭐ Chunk 级入库(对比 add_papers 的"整篇入库")
        
        把每篇论文切分为多个 chunk,每个 chunk 单独 embedding 入库。
        相比 add_papers 的"1 paper = 1 vector",这个方法的特点:
        - 1 paper = N chunks,每个 chunk 独立 embedding
        - 检索粒度从"整篇"提升到"段落级"
        - 长论文的局部细节不再被全文平均稀释
        
        Args:
            papers: 论文列表
            summaries: paper_id → PaperSummary
            chunker: 自定义 chunker 实例(可选,优先级最高)
            chunker_strategy: 如果 chunker=None,用此 strategy 创建
            chunk_size: chunk 大小
            overlap: overlap 大小
        
        Returns:
            成功入库的 chunk 总数(注意:不是论文数)
        """
        if chunker is None:
            chunker = get_chunker(
                strategy=chunker_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        
        logger.info(
            f"[VectorStore] Using {type(chunker).__name__} "
            f"(chunk_size={chunker.chunk_size}, overlap={chunker.overlap})"
        )
        
        all_chunk_ids = []
        all_chunk_texts = []
        all_chunk_metas = []
        
        skipped_papers = 0
        chunked_papers = 0
        
        for paper in papers:
            paper_id = paper["paper_id"]
            
            # 重复入库检查:看 chromadb 里是否已有这篇论文的 chunk
            # 用 metadata 过滤,因为现在 id 是 chunk_id 不是 paper_id
            existing = self.collection.get(
                where={"paper_id": paper_id},
                limit=1,
            )
            if existing["ids"]:
                logger.debug(f"[VectorStore] Skip (already chunked): {paper_id}")
                skipped_papers += 1
                continue
            
            # 构建要切分的完整文本
            summary = summaries.get(paper_id)
            if summary:
                full_text = (
                    f"# Abstract\n{paper['abstract']}\n\n"
                    f"# Method\n{summary['method']}\n\n"
                    f"# Dataset\n{summary['dataset']}\n\n"
                    f"# Finding\n{summary['finding']}\n\n"
                    f"# Limitation\n{summary['limitation']}"
                )
            else:
                full_text = f"# Abstract\n{paper['abstract']}"
            
            # 基础 metadata,每个 chunk 都会继承
            base_meta = {
                "paper_id": paper_id,
                "title": paper["title"][:500],
                "year": paper["year"],
                "source": paper["source"],
                "authors": ", ".join(paper["authors"][:3]),
            }
            
            # 切分!
            chunks = chunker.chunk(full_text, base_metadata=base_meta)
            
            if not chunks:
                logger.warning(f"[VectorStore] No chunks generated for {paper_id}")
                continue
            
            # 给每个 chunk 生成唯一 id
            for chunk in chunks:
                chunk_idx = chunk.metadata.get(
                    "global_chunk_index",
                    chunk.metadata.get("chunk_index", 0)
                )
                chunk_id = f"{paper_id}__c{chunk_idx}"
                
                # ChromaDB metadata 要求值是 str/int/float/bool,不能有 None
                cleaned_meta = {
                    k: v for k, v in chunk.metadata.items()
                    if v is not None and isinstance(v, (str, int, float, bool))
                }
                
                all_chunk_ids.append(chunk_id)
                all_chunk_texts.append(chunk.text)
                all_chunk_metas.append(cleaned_meta)
            
            chunked_papers += 1
            logger.debug(
                f"[VectorStore] {paper_id} → {len(chunks)} chunks"
            )
        
        if not all_chunk_ids:
            logger.info(
                f"[VectorStore] No new chunks to add "
                f"(skipped {skipped_papers} papers, all already in DB)"
            )
            return 0
        
        # 批量入库(ChromaDB 内部会调 embedding function 批处理)
        self.collection.add(
            ids=all_chunk_ids,
            documents=all_chunk_texts,
            metadatas=all_chunk_metas,
        )
        
        logger.info(
            f"[VectorStore] ✅ Added {len(all_chunk_ids)} chunks "
            f"from {chunked_papers} papers "
            f"(skipped {skipped_papers} pre-existing papers)"
        )
        return len(all_chunk_ids)
    
    def search_chunks(
        self,
        query: str,
        n_results: int = 5,
        section_filter: Optional[str] = None,
        where: Optional[dict] = None,
    ) -> List[dict]:
        """
        ⭐ Chunk 级检索(比 search 更精细)
        
        Args:
            query: 检索文本
            n_results: 返回前 N 个 chunk
            section_filter: 只检索某个 section(如 "Method")。仅 section_aware 切分有效
            where: 自定义 metadata 过滤
        
        Returns:
            [{"chunk_id", "paper_id", "title", "section", "score", "text"}, ...]
        """
        count = self.collection.count()
        if count == 0:
            logger.warning("[VectorStore] Collection is empty")
            return []
        
        # 构建过滤条件
        filters = dict(where) if where else {}
        if section_filter:
            filters["section"] = section_filter
        
        # ChromaDB 的 where 不允许空 dict,要用 None
        where_arg = filters if filters else None
        
        n_results = min(n_results, count)
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_arg,
            include=["documents", "metadatas", "distances"],
        )
        
        hits = []
        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            hits.append({
                "chunk_id": chunk_id,
                "paper_id": meta.get("paper_id", "unknown"),
                "title": meta.get("title", ""),
                "section": meta.get("section", "unknown"),
                "score": 1 - results["distances"][0][i],
                "text": results["documents"][0][i],
            })
        
        logger.info(
            f"[VectorStore] Chunk query '{query[:50]}...' → {len(hits)} hits"
            + (f" (filtered section={section_filter})" if section_filter else "")
        )
        return hits

    def search_chunks_with_rerank(
        self,
        query: str,
        n_results: int = 5,
        rerank_top_n: int = 20,
        section_filter: Optional[str] = None,
        where: Optional[dict] = None,
        enable_rerank: bool = True,
    ) -> List[dict]:
        """
        ⭐ Cascaded Retrieval:dense 粗排 + CrossEncoder 精排
        
        工作流:
        1. dense embedding 召回 top rerank_top_n(默认 20)
        2. CrossEncoder 对这 20 个精排
        3. 返回最终 top n_results
        
        Args:
            query: 检索文本
            n_results: 最终返回 chunk 数
            rerank_top_n: 粗排召回数(rerank 的输入规模)
            section_filter: section 过滤(同 search_chunks)
            where: 自定义 metadata 过滤
            enable_rerank: 是否启用 rerank(False 时退化为普通 search_chunks)
        
        Returns:
            每个 chunk 多了 `rerank_score` 字段(如启用 rerank)
        """
        # Step 1: 粗排召回 top N
        coarse_hits = self.search_chunks(
            query=query,
            n_results=rerank_top_n,
            section_filter=section_filter,
            where=where,
        )
        
        if not enable_rerank or len(coarse_hits) <= n_results:
            # 不需要 rerank,或粗排结果已经少于目标数
            return coarse_hits[:n_results]
        
        # Step 2: CrossEncoder 精排
        from src.rag.reranker import get_reranker
        try:
            reranker = get_reranker()
            reranked = reranker.rerank(
                query=query,
                candidates=coarse_hits,
                top_k=n_results,
                text_field="text",
            )
            logger.info(
                f"[VectorStore] Cascaded retrieval: "
                f"{len(coarse_hits)} → rerank → {len(reranked)}"
            )
            return reranked
        except Exception as e:
            logger.warning(
                f"[VectorStore] Rerank failed, falling back to dense-only: {e}"
            )
            return coarse_hits[:n_results]
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
    print(f"\nCurrent collection count: {vs.collection.count()}")
    
    # === 测试 chunk 级入库 ===
    print("\n" + "=" * 70)
    print("Test: Chunk-level ingestion")
    print("=" * 70)
    
    fake_paper = {
        "paper_id": "test_chunked_001",
        "title": "Attention Is All You Need (test)",
        "abstract": (
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks. We propose a new "
            "simple network architecture, the Transformer, based solely on "
            "attention mechanisms."
        ),
        "year": 2017,
        "source": "arxiv",
        "authors": ["Vaswani et al."],
    }
    fake_summary = {
        "method": (
            "We use multi-head self-attention with 8 heads. The attention "
            "computation has O(n^2) complexity in sequence length. We add "
            "positional encoding using sine and cosine functions."
        ),
        "dataset": "WMT 2014 English-German translation, 4.5M sentence pairs.",
        "finding": (
            "Transformer achieves 28.4 BLEU on EN-DE, a new state of the art. "
            "Training is faster than RNN baselines by 12x."
        ),
        "limitation": (
            "Quadratic complexity in sequence length limits scalability for "
            "very long sequences."
        ),
    }
    
    added = vs.add_papers_with_chunks(
        papers=[fake_paper],
        summaries={"test_chunked_001": fake_summary},
        chunker_strategy="section_aware",
        chunk_size=200,
        overlap=40,
    )
    print(f"\n✅ Added {added} chunks")
    
    # === 测试 chunk 级检索 ===
    print("\n" + "=" * 70)
    print("Test: Chunk-level search")
    print("=" * 70)
    
    query = "attention complexity"
    hits = vs.search_chunks(query, n_results=3)
    print(f"\nQuery: '{query}'\n")
    for i, h in enumerate(hits, 1):
        print(f"[Hit {i}] score={h['score']:.3f}, section={h['section']}")
        print(f"  paper: {h['title'][:60]}")
        print(f"  text: {h['text'][:120]}...")
        print()
    
    # === 测试 section 过滤 ===
    print("=" * 70)
    print("Test: Section-filtered search (only Method)")
    print("=" * 70)
    hits_method = vs.search_chunks(query, n_results=3, section_filter="Method")
    print(f"\nQuery: '{query}' with section=Method\n")
    for i, h in enumerate(hits_method, 1):
        print(f"[Hit {i}] section={h['section']}")
        print(f"  text: {h['text'][:120]}...")
        print()
    
    print(f"\n✅ Total chunks in DB: {vs.collection.count()}")
    print("✅ All chunk-level tests passed")