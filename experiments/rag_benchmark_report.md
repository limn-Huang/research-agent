# RAG 配置对比实验报告

## 实验设计

- 数据集:当前 ChromaDB 中已索引的论文(由 `python main.py` 入库)
- 查询数:3 个不同方向的 query
- 评估指标:
  - **Top-1 Informative Rate**:Top-1 返回 Method/Result/Finding 章节的比例(越高越好)
  - **Top-5 Informative Count**:Top-5 中信息密集 section 的平均数(越高越好)
  - **Informative Avg Rank**:信息密集 section 的平均排名(越低越好)

## 实验结果对比

| 配置 | 描述 | Top-1 Informative Rate | Top-5 Informative Count | Informative Avg Rank |
|------|------|----------------------|------------------------|----------------------|
| A_baseline | Baseline (no chunk, no rerank) | 0.0% | 0.00 | 0.00 |
| B_sliding | Sliding only | 66.7% | 3.67 | 2.92 |
| C_section_aware | Section-aware only | 66.7% | 3.67 | 2.92 |
| D_section_rerank | Section-aware + Rerank | 66.7% | 4.00 | 3.17 |

## 关键洞察

1. **Section-aware chunker 自身已带来 section 级精度** — Top-5 informative count 达 3.7/5

2. **Rerank 在 Top-1 精度上的提升** — 从 67% 提升到 67%(+0 pp)

3. **Informative section 平均排名提前** — 从 2.92 提升到 3.17(--0.25)
