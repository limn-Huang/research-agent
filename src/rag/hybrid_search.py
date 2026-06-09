"""
Hybrid Search:BM25(关键词) + Dense(语义) 混合检索。

为什么要 Hybrid?
- Dense(向量检索):擅长语义相似,但对精确词匹配弱
  例:"carbon footprint" 能匹配到"CO2 emission"(语义近)
- BM25(关键词检索):擅长精确词匹配,但不懂语义
  例:能精确找到含"carbon"的文章

混合两者:
  Final Score = α × dense_score + (1-α) × bm25_score
  α=0.7 时偏语义,α=0.3 时偏关键词

实际效果:Hybrid 比单独用任一种,Recall 通常提升 10-20%
"""

import logging
import math
from collections import Counter
from typing import List, Dict

logger = logging.getLogger(__name__)


class BM25:
    """
    BM25 算法实现。

    BM25 是 TF-IDF 的改进版:
    - TF(词频):词出现越多越相关,但有上限(超过一定次数边际收益递减)
    - IDF(逆文档频率):罕见词比常见词更有区分度
    - k1, b 是调节参数(通常 k1=1.5, b=0.75)

    不需要完全理解数学,记住这句话:
    "BM25 = 改进版 TF-IDF,衡量文档和 query 的关键词匹配程度"
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []          # tokenized 文档列表
        self.doc_ids: List[str] = []             # 对应的 paper_id
        self.doc_freq: Dict[str, int] = {}       # 每个词出现在几个文档里
        self.avg_doc_len: float = 0.0

    def add_documents(self, doc_ids: List[str], documents: List[str]):
        """添加文档到索引"""
        for doc_id, doc in zip(doc_ids, documents):
            tokens = self._tokenize(doc)
            self.docs.append(tokens)
            self.doc_ids.append(doc_id)
            # 更新 doc_freq
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        total_len = sum(len(d) for d in self.docs)
        self.avg_doc_len = total_len / len(self.docs) if self.docs else 0

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """BM25 检索,返回 top-n 结果"""
        if not self.docs:
            return []

        query_tokens = self._tokenize(query)
        n = len(self.docs)
        scores = []

        for idx, doc_tokens in enumerate(self.docs):
            score = 0.0
            doc_len = len(doc_tokens)
            tf_counter = Counter(doc_tokens)

            for token in query_tokens:
                tf = tf_counter.get(token, 0)
                df = self.doc_freq.get(token, 0)
                if df == 0:
                    continue

                # BM25 公式
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                )
                score += idf * tf_norm

            scores.append((self.doc_ids[idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"paper_id": pid, "bm25_score": s}
            for pid, s in scores[:n_results]
            if s > 0
        ]

    def _tokenize(self, text: str) -> List[str]:
        """简单分词:小写 + 按空格/标点分割"""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z]{2,}\b', text)
        return tokens


def hybrid_search(
    query: str,
    vector_store,
    bm25_index: BM25,
    all_paper_ids: List[str],
    n_results: int = 5,
    alpha: float = 0.7,
) -> List[Dict]:
    """
    Hybrid Search 主函数。

    Args:
        query: 检索问题
        vector_store: PaperVectorStore 实例
        bm25_index: BM25 实例
        all_paper_ids: 所有论文 ID(用于对齐两种检索的结果)
        n_results: 最终返回几篇
        alpha: dense 权重(0~1),1-alpha 是 BM25 权重

    Returns:
        融合排序后的 top-n 论文 ID 列表
    """
    # === Dense 检索 ===
    dense_results = vector_store.search(query, n_results=n_results * 2)
    dense_scores = {r["paper_id"]: r["score"] for r in dense_results}

    # === BM25 检索 ===
    bm25_results = bm25_index.search(query, n_results=n_results * 2)
    bm25_scores = {r["paper_id"]: r["bm25_score"] for r in bm25_results}

    # === Normalize BM25 分数到 [0, 1] ===
    if bm25_scores:
        max_bm25 = max(bm25_scores.values())
        if max_bm25 > 0:
            bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}

    # === 融合 ===
    all_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
    combined = []
    for pid in all_ids:
        d_score = dense_scores.get(pid, 0.0)
        b_score = bm25_scores.get(pid, 0.0)
        final_score = alpha * d_score + (1 - alpha) * b_score
        combined.append({"paper_id": pid, "score": final_score,
                         "dense_score": d_score, "bm25_score": b_score})

    combined.sort(key=lambda x: x["score"], reverse=True)

    logger.info(f"[HybridSearch] Query: '{query[:50]}' → {len(combined[:n_results])} results")
    for r in combined[:n_results]:
        logger.debug(
            f"  {r['paper_id']}: final={r['score']:.3f} "
            f"(dense={r['dense_score']:.3f}, bm25={r['bm25_score']:.3f})"
        )

    return combined[:n_results]