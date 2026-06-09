"""
Retriever 基类 —— 信息聚合 Agent 框架的核心扩展点。

设计理念:
所有数据源(arXiv、Semantic Scholar、Web Search、内部知识库...)都实现这个接口。
这样的好处:
1. 切换数据源只换一行代码(BaseRetriever 的多态)
2. 加新数据源不需要改 Agent 代码,只要写一个新的 Retriever 类
3. 测试时可以用 MockRetriever 替换真实 API
4. 面试时可以指着说:"这就是 Strategy 设计模式 + 依赖注入"
"""

from abc import ABC, abstractmethod
from typing import List
from src.state import Paper


class BaseRetriever(ABC):
    """
    所有 Retriever 的抽象基类。
    
    ABC (Abstract Base Class):Python 的抽象基类机制
    - 用 @abstractmethod 装饰的方法,子类必须实现
    - 防止有人忘记实现某个方法
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据源的名字,如 'arxiv'、'web'"""
        pass
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """
        搜索数据源,返回结构化结果。
        
        所有 retriever 都必须返回 Paper 格式 —— 这是 retriever 和 Agent
        之间的"契约",保证后续节点不需要关心数据从哪来。
        """
        pass


def get_retriever(source: str) -> BaseRetriever:
    """
    Retriever 工厂函数。
    
    根据数据源名字返回对应的 Retriever 实例。
    Day 7 加新数据源时,只要在这里加一个 elif 即可。
    """
    # 延迟 import 避免循环依赖
    from src.retrievers.arxiv_retriever import ArxivRetriever
    
    if source == "arxiv":
        return ArxivRetriever()
    # Day 7 添加:
    # elif source == "web":
    #     return WebRetriever()
    else:
        raise ValueError(f"Unknown retriever source: {source}")