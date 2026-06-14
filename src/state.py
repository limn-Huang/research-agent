"""
ResearchState - LangGraph 的核心状态对象

设计原则:
1. 扁平 > 嵌套(方便调试)
2. 所有字段必须可 JSON 序列化(因为要被 checkpoint)
3. 每个字段标注由哪个节点负责更新
4. 中间状态字段用 default 值,避免 KeyError
"""

from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add


class Paper(TypedDict):
    """单篇论文的元数据"""
    paper_id: str           # arXiv ID 或 SemanticScholar ID
    title: str
    authors: List[str]
    abstract: str
    year: int
    pdf_url: Optional[str]  # PDF 下载地址(可能没有)
    source: str             # "arxiv" / "semantic_scholar"


class PaperSummary(TypedDict):
    """Reading Agent 提取的结构化摘要"""
    paper_id: str
    method: str             # 用了什么方法
    dataset: str            # 用了什么数据
    finding: str            # 主要发现
    limitation: str         # 论文自陈或我们识别的局限


class SubTask(TypedDict):
    """Planner 拆出来的子任务"""
    task_id: str
    description: str
    status: str             # "pending" / "running" / "done" / "failed"


class ResearchState(TypedDict):
    """
    整个 graph 共享的状态。
    
    数据流(节点更新责任):
        Planner   → sub_tasks
        Retrieval → papers
        Reading   → paper_summaries
        Comparison → method_comparison_table
        Reporter  → final_report
    """
    
    # === 用户输入(初始化时设置)===
    query: str                                  # 原始研究问题
    max_papers: int                             # 最多检索几篇论文(成本控制)
    
    # === Planner 输出(升级) ===
    sub_tasks: List[SubTask]
    search_query_en: str     # ← 新增:英文检索词(Planner 生成)

    # === RAG 相关 ===
    retrieved_context: List[str]   # ← 新增:RAG 检索出的相关 chunk(Reporter 用)
    
    # === Retrieval 输出 ===
    papers: List[Paper]                         # 检索到的论文列表
    
    # === Reading 输出 ===
    paper_summaries: Dict[str, PaperSummary]    # paper_id -> 结构化摘要
    
    # === Comparison 输出 ===
    method_comparison_table: str                # markdown 格式的对比表
    
    # === Reporter 输出 ===
    final_report: str                           # 最终综述
    
    # === 元信息(可观测性 & debug 用)===
    # messages 用 Annotated[..., add] 表示:多个节点写入时自动追加而非覆盖
    messages: Annotated[List[str], add]         # 节点的日志消息(累积)
    error: Optional[str]                        # 出错信息(任意节点可以写)
    step_count: int                             # 已执行的步骤数(防死循环)
    
    #Chunker配置
    chunker_strategy:str    #"fixed"/"sliding"/"section_aware"
    chunk_size:int          # 每个 chunk 的文本长度(字符数)默认 512
    overlap:int             # 相邻 chunk 之间的重叠字符数  默认 80


def create_initial_state(query: str, max_papers: int = 10) -> ResearchState:
    """
    创建初始 State。
    
    为什么需要这个函数:
    - LangGraph 不要求所有字段必须有值,但**显式初始化能避免 KeyError 排查地狱**
    - 集中管理默认值
    """
    return ResearchState(
        query=query,
        max_papers=max_papers,
        sub_tasks=[],
        search_query_en="",
        retrieved_context=[],
        papers=[],
        paper_summaries={},
        method_comparison_table="",
        final_report="",
        messages=[],
        error=None,
        step_count=0,
    )