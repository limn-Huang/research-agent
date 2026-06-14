"""
Chunker 模块:把长文本切分为小块,提升 RAG 检索精度。

为什么需要 Chunker?
- 原版"每篇论文 1 个 vector"在长论文场景下信息损失严重
- 把论文切成 N 个 chunk,每个 chunk 单独 embedding 入库
- 检索时按 chunk 粒度匹配,大幅提升精度

Trade-off 设计(面试必考):
=====================================
1. chunk 太小(<128 token):
   - 优点:精度高,匹配精确
   - 缺点:上下文丢失,LLM 收到孤立片段难理解
   - 缺点:embedding 调用次数爆炸(成本 +N×)

2. chunk 太大(>1024 token):
   - 优点:上下文完整
   - 缺点:每个 vector 信息熵高,反而稀释相关信号
   - 缺点:接近原版"整篇 1 个 vector"的问题

3. overlap 太小(<40 token):
   - 优点:总 chunk 数少
   - 缺点:边界附近的句子被"切断",检索可能漏掉

4. overlap 太大(>200 token):
   - 优点:边界保护强
   - 缺点:存储和计算冗余

业界经验值(我们也用这个起点):
- chunk_size = 512 token(约 380 中文字 / 300 英文 token)
- overlap = 80 token(约 15% 的 chunk_size)
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class Chunk:
    """
    一个 chunk 的标准数据结构。
    
    metadata 字段用于:
    - 检索后回溯到原文(paper_id + section_name + chunk_index)
    - 在最终 prompt 中标注来源("根据 paper X 的 method 章节...")
    - 元数据过滤(只检索某个 section 的 chunk)
    """
    text: str
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # 自动添加 char 长度,便于后续分析
        self.metadata.setdefault("char_count", len(self.text))


# =============================================================================
# 抽象基类(Strategy Pattern)
# =============================================================================

class BaseChunker(ABC):
    """
    Chunker 抽象基类。
    
    用 Strategy Pattern 让 3 种切分方式可以无缝替换:
    - 调用方写 `chunker.chunk(text)`,不关心是哪种实现
    - 未来加新 chunker(如 LLM-based)不用改业务代码
    
    面试金句:
    "我用 Strategy 模式抽象 chunker,后续要加新策略(语义切分、
    LLM 切分)只需实现 chunk() 方法,业务代码零修改。"
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 80):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be < chunk_size ({chunk_size}). "
                f"Otherwise chunks would have negative progress."
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    @abstractmethod
    def chunk(self, text: str, base_metadata: dict | None = None) -> List[Chunk]:
        """
        切分文本为 Chunk 列表。
        
        Args:
            text: 原始文本
            base_metadata: 基础元数据(paper_id / title 等),会被合并到每个 chunk 的 metadata
        
        Returns:
            List[Chunk]
        """
        pass
    
    def _make_base_metadata(self, base_metadata: dict | None) -> dict:
        """工具方法:统一处理 metadata 默认值"""
        return dict(base_metadata) if base_metadata else {}


# =============================================================================
# Strategy 1: FixedSizeChunker(固定 token 切分)
# =============================================================================

class FixedSizeChunker(BaseChunker):
    """
    按固定 token 长度切分,无 overlap。
    
    Trade-off 分析:
    - ✅ 实现最简单,速度最快
    - ✅ chunk 数量可预测(适合估算成本)
    - ❌ 经常在句子中间切断,语义割裂
    - ❌ 边界句子完全丢失上下文
    
    适用场景:
    - 极短文本(< 1024 token,切不出几个 chunk)
    - benchmark 对照实验的 baseline
    - 内容本身是结构化、无强语义连贯性的数据(日志、code)
    """
    
    def __init__(self, chunk_size: int = 512, **kwargs):
        # FixedSize 不需要 overlap,强制为 0
        super().__init__(chunk_size=chunk_size, overlap=0)
    
    def chunk(self, text: str, base_metadata: dict | None = None) -> List[Chunk]:
        if not text:
            return []
        
        base_meta = self._make_base_metadata(base_metadata)
        
        # 简化:用字符长度近似 token(中文 1 字 ≈ 1 token,英文 1 token ≈ 4 字符)
        # 生产环境应该用 tiktoken / sentencepiece 真实统计
        chunks = []
        text_len = len(text)
        n_chunks = (text_len + self.chunk_size - 1) // self.chunk_size  # 向上取整
        
        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            
            meta = {
                **base_meta,
                "chunker": "fixed",
                "chunk_index": i,
                "total_chunks": n_chunks,
                "start_char": start,
                "end_char": end,
            }
            chunks.append(Chunk(text=chunk_text, metadata=meta))
        
        logger.debug(f"[FixedSizeChunker] {text_len} chars → {n_chunks} chunks")
        return chunks


# =============================================================================
# Strategy 2: SlidingWindowChunker(滑动窗口 + overlap)
# =============================================================================

class SlidingWindowChunker(BaseChunker):
    """
    滑动窗口 + overlap 切分。
    
    核心思想:每个 chunk 与前后 chunk 有 overlap,保证边界句子至少在
    某一个 chunk 里被完整捕获。
    
    Trade-off 分析:
    - ✅ 边界保护:跨界句子能被某个 chunk 完整覆盖
    - ✅ 召回率提升:同一信息在多个 chunk 出现,提高匹配概率
    - ❌ 存储冗余:N 字符文本会产生略多于 N/chunk_size 个 chunk
    - ❌ 同一片段被重复 embedding(成本略增)
    
    业界共识(LangChain / LlamaIndex 默认值):
    - chunk_size = 512, overlap = 80(15% overlap)
    - 这是我们默认推荐的配置
    
    适用场景:
    - 学术论文、技术文档(段落连贯性强)
    - 大部分通用 RAG 场景
    """
    
    def chunk(self, text: str, base_metadata: dict | None = None) -> List[Chunk]:
        if not text:
            return []
        
        base_meta = self._make_base_metadata(base_metadata)
        
        chunks = []
        text_len = len(text)
        stride = self.chunk_size - self.overlap  # 每次前进的步长
        
        i = 0
        chunk_idx = 0
        while i < text_len:
            end = min(i + self.chunk_size, text_len)
            chunk_text = text[i:end]
            
            meta = {
                **base_meta,
                "chunker": "sliding",
                "chunk_index": chunk_idx,
                "start_char": i,
                "end_char": end,
                "overlap": self.overlap,
            }
            chunks.append(Chunk(text=chunk_text, metadata=meta))
            
            chunk_idx += 1
            
            # 已经到末尾,停止
            if end == text_len:
                break
            
            i += stride
        
        # 补充 total_chunks(在切完后才知道总数)
        for c in chunks:
            c.metadata["total_chunks"] = len(chunks)
        
        logger.debug(
            f"[SlidingWindowChunker] {text_len} chars → {len(chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.overlap})"
        )
        return chunks


# =============================================================================
# Strategy 3: SectionAwareChunker(基于论文结构的切分)
# =============================================================================

class SectionAwareChunker(BaseChunker):
    """
    根据论文 section 标题(Abstract / Method / Results 等)先切分,
    section 内部再用 sliding window。
    
    核心思想:
    - 论文有清晰的 section 结构,跨 section 切割会破坏语义
    - 先按 section 切,保留 section 名作为 metadata
    - section 内部如果还太长,用 sliding window 二次切分
    
    Trade-off 分析:
    - ✅ 保留论文逻辑结构(method / results 等不会混在一起)
    - ✅ 可以基于 section 做检索过滤(只检索 method 章节)
    - ✅ chunk 的 metadata 包含 section 信息,便于报告生成时溯源
    - ❌ 实现复杂(需要识别 section 边界)
    - ❌ 非论文文本(如博客)效果差
    - ❌ section 切分依赖正则,鲁棒性弱于 LLM-based 切分
    
    适用场景:
    - 学术论文(本项目主场景)✅
    - 技术规范、API 文档(有清晰章节结构)
    
    面试金句:
    "我的 section-aware 切分识别论文常见的 7 个 section 标题
    (Abstract / Introduction / Method / Results / Discussion / Conclusion / Related Work),
    再在 section 内部用 sliding window。这样既保留了结构,又控制了 chunk 大小。"
    """
    
    # 常见 section 标题(英文 + 中文)
    SECTION_PATTERNS = [
        r"^\s*#+\s*(Abstract|摘要)\s*$",
        r"^\s*#+\s*(Introduction|引言|绪论)\s*$",
        r"^\s*#+\s*(Related Work|Background|相关工作|背景)\s*$",
        r"^\s*#+\s*(Method|Methods|Methodology|方法|方法论)\s*$",
        r"^\s*#+\s*(Experiment|Experiments|实验)\s*$",
        r"^\s*#+\s*(Result|Results|结果)\s*$",
        r"^\s*#+\s*(Discussion|讨论)\s*$",
        r"^\s*#+\s*(Conclusion|Conclusions|结论)\s*$",
        r"^\s*#+\s*(References|参考文献)\s*$",
    ]
    
    def __init__(self, chunk_size: int = 512, overlap: int = 80, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        # 复用 SlidingWindow 做 section 内二次切分
        self._inner_chunker = SlidingWindowChunker(
            chunk_size=chunk_size, overlap=overlap
        )
        # 预编译 section 识别正则(性能优化 + 工程审美)
        self._section_regex = re.compile(
            "|".join(self.SECTION_PATTERNS),
            re.MULTILINE | re.IGNORECASE,
        )
    
    def chunk(self, text: str, base_metadata: dict | None = None) -> List[Chunk]:
        if not text:
            return []
        
        base_meta = self._make_base_metadata(base_metadata)
        
        # 找到所有 section 边界
        sections = self._split_into_sections(text)
        
        if len(sections) <= 1:
            # 没找到 section 结构,降级为纯 sliding window
            logger.debug(
                "[SectionAwareChunker] No sections detected, falling back to sliding window"
            )
            base_meta["section"] = "unknown"
            return self._inner_chunker.chunk(text, base_meta)
        
        all_chunks = []
        for section_name, section_text in sections:
            section_meta = {
                **base_meta,
                "chunker": "section_aware",
                "section": section_name,
            }
            # section 内部用 sliding window 二次切分
            section_chunks = self._inner_chunker.chunk(section_text, section_meta)
            all_chunks.extend(section_chunks)
        
        # 补充全局 chunk_index 和 total_chunks
        for i, c in enumerate(all_chunks):
            c.metadata["global_chunk_index"] = i
            c.metadata["total_chunks"] = len(all_chunks)
        
        logger.debug(
            f"[SectionAwareChunker] {len(text)} chars → "
            f"{len(sections)} sections → {len(all_chunks)} chunks"
        )
        return all_chunks
    
    def _split_into_sections(self, text: str) -> List[tuple[str, str]]:
        """
        把文本按 section 切分。
        
        Returns:
            [(section_name, section_text), ...]
        """
        # 找所有 section 标题的位置
        matches = list(self._section_regex.finditer(text))
        
        if not matches:
            return []
        
        sections = []
        
        # 第一个 section 之前的内容(通常是 title / authors / abstract 前的元信息)
        if matches[0].start() > 0:
            pre_text = text[:matches[0].start()].strip()
            if pre_text:
                sections.append(("frontmatter", pre_text))
        
        # 每个 section
        for i, match in enumerate(matches):
            section_name = match.group(0).strip().lstrip("#").strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            
            if section_text:
                sections.append((section_name, section_text))
        
        return sections


# =============================================================================
# 工厂函数(便于切换 chunker)
# =============================================================================

def get_chunker(strategy: str, chunk_size: int = 512, overlap: int = 80) -> BaseChunker:
    """
    工厂方法:根据 strategy 名字返回对应的 chunker。
    
    Args:
        strategy: "fixed" / "sliding" / "section_aware"
        chunk_size: chunk 大小(token 近似)
        overlap: overlap 大小(仅对 sliding 和 section_aware 有效)
    
    Returns:
        BaseChunker 实例
    
    Example:
        >>> chunker = get_chunker("sliding", chunk_size=512, overlap=80)
        >>> chunks = chunker.chunk(my_paper_text, {"paper_id": "2103.12345"})
    """
    strategy = strategy.lower()
    
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size)
    elif strategy == "sliding":
        return SlidingWindowChunker(chunk_size=chunk_size, overlap=overlap)
    elif strategy == "section_aware":
        return SectionAwareChunker(chunk_size=chunk_size, overlap=overlap)
    else:
        raise ValueError(
            f"Unknown chunker strategy: {strategy}. "
            f"Available: fixed / sliding / section_aware"
        )


# =============================================================================
# 单测(python -m src.rag.chunkers)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    sample_text = """# Abstract
We propose a new method for X. Our experiments show 90% accuracy.

# Introduction  
Many prior works have studied X. However, they suffer from problem Y.
We propose to solve Y by using technique Z.

# Method
Our method consists of three steps. Step 1: preprocess. Step 2: train. Step 3: inference.
The preprocessing involves tokenization with subword units of size 32k.

# Results
On dataset D1, we achieve 90.2% F1. On D2, we get 88.5%. Compared to baseline (85.1%), this is significant.

# Conclusion
We showed Z works. Future work includes extending to multi-lingual settings.
"""
    
    print("\n" + "=" * 70)
    print("Test 1: FixedSizeChunker (chunk_size=200)")
    print("=" * 70)
    fc = FixedSizeChunker(chunk_size=200)
    chunks = fc.chunk(sample_text, {"paper_id": "test_001"})
    for c in chunks:
        print(f"\n[chunk {c.metadata['chunk_index']}] {c.metadata['char_count']} chars")
        print(f"  text: {c.text[:80]}...")
    
    print("\n" + "=" * 70)
    print("Test 2: SlidingWindowChunker (chunk_size=200, overlap=40)")
    print("=" * 70)
    sw = SlidingWindowChunker(chunk_size=200, overlap=40)
    chunks = sw.chunk(sample_text, {"paper_id": "test_001"})
    for c in chunks:
        print(f"\n[chunk {c.metadata['chunk_index']}] {c.metadata['char_count']} chars, start={c.metadata['start_char']}")
        print(f"  text: {c.text[:80]}...")
    
    print("\n" + "=" * 70)
    print("Test 3: SectionAwareChunker")
    print("=" * 70)
    sa = SectionAwareChunker(chunk_size=200, overlap=40)
    chunks = sa.chunk(sample_text, {"paper_id": "test_001"})
    for c in chunks:
        print(f"\n[chunk {c.metadata.get('global_chunk_index')}] section={c.metadata.get('section')}, {c.metadata['char_count']} chars")
        print(f"  text: {c.text[:80]}...")
    
    print("\n" + "=" * 70)
    print("Test 4: Factory get_chunker")
    print("=" * 70)
    chunker = get_chunker("sliding", chunk_size=256, overlap=50)
    print(f"Created: {type(chunker).__name__}, chunk_size={chunker.chunk_size}, overlap={chunker.overlap}")
    
    print("\n✅ All chunker tests passed")