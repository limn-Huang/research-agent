# Multi-Source Information Aggregation Agent

基于 LangGraph 构建的多源信息聚合 Agent 框架,通用解决"检索 → 提取 → 对比 → 综合"类任务。论文场景为 demo,可零修改迁移到竞品分析、行业研究等领域。

## 核心特性

- **5 节点 StateGraph**:Planner / Retrieval / Reading / Comparison / Reporter
- **数据源插件化**:基于 Strategy 模式的 BaseRetriever 抽象,支持 arXiv / Web Search 等
- **Hybrid RAG**:BM25 关键词检索 + Dense Embedding 语义检索 + 加权融合
- **结构化提取**:LLM 严格 JSON 输出 + 容错降级
- **跨语言检索**:Planner 自动将中文 query 翻译为英文检索关键词

## RAG 工程

### Chunker(Strategy Pattern)
3 种切分策略可切换,统一抽象基类 `BaseChunker`:

| 策略 | 适用场景 |
|---|---|
| `FixedSizeChunker` | 极短文本、benchmark baseline |
| `SlidingWindowChunker` | 通用 RAG(512 token + 80 overlap) |
| `SectionAwareChunker` | 学术论文 ⭐(识别 Abstract/Method/Result 等 7 个 section) |

```python
from src.rag.chunkers import get_chunker
chunker = get_chunker("section_aware", chunk_size=512, overlap=80)
chunks = chunker.chunk(paper_text, {"paper_id": "..."})
```

### CrossEncoder Rerank
Cascaded Retrieval:Dense embedding 召回 Top 15 → CrossEncoder 精排 Top 5

- **模型**:`BAAI/bge-reranker-base`(280MB,中英双语)
- **实测效果**:在多关键词 query("attention complexity")上,
  Method chunk 的 rerank_score 从 dense 时代的 0.521 提升到 **0.945**,
  修正了 dense embedding 被高频词主导的问题

### Benchmark
`experiments/rag_benchmark.py` 自建对照实验,量化各组件 trade-off。
完整报告:[`experiments/rag_benchmark_report.md`](experiments/rag_benchmark_report.md)

## 架构
用户 query
↓
┌─────────┐
│ Planner │ 拆解任务 + 生成英文检索词
└────┬────┘
↓
┌──────────┐
│Retrieval │ 多源插件化检索 (BaseRetriever)
└────┬─────┘
↓
┌─────────┐
│ Reading │ PDF 解析 + LLM 结构化提取 (method/dataset/finding/limitation)
└────┬────┘
↓
┌────────────┐
│ Comparison │ 对比表 + Research Gap 识别 + 向量入库
└────┬───────┘
↓
┌──────────┐
│ Reporter │ RAG 增强 + 综述生成
└──────────┘
## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 配置 .env
cp .env.example .env
# 编辑 .env 填入 ZHIPUAI_API_KEY

# 3. 运行
python main.py
```

## 技术栈

- **Agent 编排**:LangGraph
- **LLM**:GLM-5.1(智谱)
- **向量库**:ChromaDB
- **检索**:BM25 + Dense Embedding(embedding-3)
- **PDF 处理**:PyMuPDF
- **数据源**:arXiv API

## 项目结构
src/
├── state.py              # ResearchState 定义
├── llm.py                # GLM 客户端
├── graph.py              # LangGraph 主图
├── agents/               # 5 个 Agent 节点
├── retrievers/           # 数据源插件
│   ├── base.py           # BaseRetriever 抽象
│   └── arxiv_retriever.py
├── rag/                  # RAG 模块
│   ├── vector_store.py
│   └── hybrid_search.py
└── utils/
└── pdf_processor.py



## License

MIT