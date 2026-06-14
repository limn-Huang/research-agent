"""
Reranker 模块:用 CrossEncoder 对粗排结果做精排。

为什么需要 Rerank?
============================
Dense Embedding 检索的根本局限:
- 用整段文本一次性映射到 fixed-size vector
- query 和 doc 各自独立 encoding,然后算相似度
- 缺乏 query-doc 之间的 fine-grained attention

CrossEncoder 的核心区别:
- query + doc 拼接后一起送给模型
- 模型对 [query, doc] 做 full self-attention
- 能捕捉 token 级别的精确匹配
- 但因为每个 query-doc pair 都要单独跑一次模型,延迟高

工程范式:Cascaded Retrieval(两阶段)
=====================================
Stage 1 (粗排):Dense embedding 召回 Top 20
              - 单次 embedding,余弦相似度,O(N) 但 N 是数据库大小
              - 速度优势:1 次 embedding 调用 + N 次余弦距离计算

Stage 2 (精排):CrossEncoder 对 Top 20 重排,返回 Top 5
              - K 次模型前向(K=20),O(K)
              - 精度优势:看到 query 和 doc 的完整交互

最终延迟:~ 1 次 embedding API + 1 次 CrossEncoder 前向(20 doc 一批)
最终精度:接近"用 LLM 给所有 doc 打分"的水平,成本只有 1/100

Trade-off 总结(面试必考):
==============================
✅ 优点:
- 精度提升明显(论文报告 +5~15% nDCG@10)
- 延迟可控(只对 Top N 重排,不动全库)
- 模型可独立部署/选型

❌ 缺点:
- 必须本地跑 CrossEncoder(下载模型,首次启动慢)
- 增加 ~50-300ms 延迟(对实时场景不友好)
- 与 dense retrieval 的 query encoding 互不复用

业界主流选择(我们也用这些起点):
- bge-reranker-large(中文最强,~560M)
- bge-reranker-base(中英双语,~280M,推荐)
- ms-marco-MiniLM-L-6-v2(英文,~80M,最快)
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# 默认模型:中英双语 + 平衡精度速度
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    """
    用 CrossEncoder 模型对 chunk 做精排。
    
    设计要点:
    - 懒加载:模型在第一次 rerank 时才下载/加载,加快冷启动
    - 单例式使用:模型加载耗时,建议全局复用
    - 失败降级:模型加载失败时,返回原排序(不阻塞主流程)
    """
    
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        self.model_name = model_name
        self._model = None  # 懒加载
    
    def _load_model(self):
        """
        懒加载 CrossEncoder 模型。
        
        首次调用会从 HuggingFace 下载模型(~280MB),
        后续从本地缓存加载(在 ~/.cache/huggingface 下)。
        """
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"[Reranker] Loading model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info(f"[Reranker] ✅ Model loaded successfully")
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from e
        except Exception as e:
            logger.error(f"[Reranker] Failed to load model: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: Optional[int] = None,
        text_field: str = "text",
    ) -> List[dict]:
        """
        对 candidates 列表做 rerank。
        
        Args:
            query: 检索 query
            candidates: 候选列表,每个 dict 必须有 `text_field` 字段
            top_k: 返回 Top K(None = 返回全部 reranked)
            text_field: 文本字段名(默认 "text",兼容 vector_store.search_chunks 输出)
        
        Returns:
            重排后的 candidates 列表,每个 dict 加了 `rerank_score` 字段
        
        Example:
            >>> hits = vs.search_chunks("attention complexity", n_results=20)
            >>> reranked = reranker.rerank("attention complexity", hits, top_k=5)
        """
        if not candidates:
            return []
        
        self._load_model()
        
        # 构建 [query, doc_text] pair 列表
        pairs = [[query, c.get(text_field, "")] for c in candidates]
        
        # 模型前向:输出每个 pair 的 score(越高越相关)
        scores = self._model.predict(pairs)
        
        # 把 score 写回 candidates,并按 score 降序排
        scored = [
            {**c, "rerank_score": float(scores[i])}
            for i, c in enumerate(candidates)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        if top_k is not None:
            scored = scored[:top_k]
        
        logger.info(
            f"[Reranker] Reranked {len(candidates)} → top-{len(scored)} "
            f"(top score: {scored[0]['rerank_score']:.3f}, "
            f"bottom: {scored[-1]['rerank_score']:.3f})"
        )
        return scored


# =============================================================================
# 全局单例(模型加载昂贵,避免重复初始化)
# =============================================================================

_global_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(model_name: str = DEFAULT_RERANK_MODEL) -> CrossEncoderReranker:
    """获取全局 Reranker 单例"""
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = CrossEncoderReranker(model_name=model_name)
    return _global_reranker


# =============================================================================
# 单测(python -m src.rag.reranker)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 模拟一组 candidates(从 vector_store 检索回来的样子)
    candidates = [
        {
            "chunk_id": "c1",
            "text": "Transformer based solely on attention mechanisms",
            "section": "Abstract",
            "score": 0.632,  # 原始 dense 分数
        },
        {
            "chunk_id": "c2",
            "text": "attention computation has O(n^2) complexity in sequence length",
            "section": "Method",
            "score": 0.521,
        },
        {
            "chunk_id": "c3",
            "text": "We propose a new simple network architecture",
            "section": "Abstract",
            "score": 0.474,
        },
        {
            "chunk_id": "c4",
            "text": "Training is faster than RNN baselines by 12x",
            "section": "Finding",
            "score": 0.412,
        },
        {
            "chunk_id": "c5",
            "text": "Quadratic complexity limits scalability for very long sequences",
            "section": "Limitation",
            "score": 0.398,
        },
    ]
    
    query = "attention complexity"
    print(f"\nQuery: '{query}'\n")
    
    print("=" * 70)
    print("Original ranking (by dense embedding score):")
    print("=" * 70)
    for i, c in enumerate(candidates, 1):
        print(f"  Rank {i}: [{c['section']}] {c['text'][:60]}...")
        print(f"            dense_score={c['score']:.3f}")
    
    # 跑 Rerank
    print("\n" + "=" * 70)
    print("Reranking with CrossEncoder...")
    print("=" * 70)
    print("(First run will download ~280MB model, please wait...)\n")
    
    reranker = get_reranker()
    reranked = reranker.rerank(query, candidates, top_k=5)
    
    print("\n" + "=" * 70)
    print("After Reranking (by CrossEncoder score):")
    print("=" * 70)
    for i, c in enumerate(reranked, 1):
        print(f"  Rank {i}: [{c['section']}] {c['text'][:60]}...")
        print(f"            dense_score={c['score']:.3f}, rerank_score={c['rerank_score']:.3f}")
    
    # 验证关键现象:Method / Limitation 应该被推到前面
    print("\n" + "=" * 70)
    print("Key observation:")
    print("=" * 70)
    top_section = reranked[0]['section']
    if top_section in ("Method", "Limitation"):
        print(f"✅ Top 1 is '{top_section}' (contains actual 'complexity' info)")
        print(f"   → CrossEncoder corrected dense embedding's bias toward Abstract")
    else:
        print(f"⚠️  Top 1 is still '{top_section}'")
        print(f"   → Either model failed to load, or this query is too noisy")
    
    print("\n✅ Reranker test complete")