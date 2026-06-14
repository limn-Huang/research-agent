"""
入口:跑一次完整研究流程。

用法:
    python main.py
"""

import logging
import sys
from pathlib import Path

from src.graph import build_graph
from src.state import create_initial_state


def setup_logging():
    """配置日志输出到 console 和文件"""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/run.log", encoding="utf-8"),
        ],
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 1. 构建 graph
    logger.info("=" * 60)
    logger.info("Building graph...")
    app = build_graph()
    
    # 2. 准备初始 state
    # 在 main() 函数里
    query = "多智能体协作框架"  # 英文 query,arXiv 检索效果更好
    initial_state = create_initial_state(query=query, max_papers=7) 
    logger.info(f"Query: {query}")
    
    # 3. 调用 graph
    logger.info("=" * 60)
    logger.info("Invoking graph...")
    result = app.invoke(initial_state)
    
    # 4. 输出结果
    logger.info("=" * 60)
    logger.info("=== EXECUTION COMPLETE ===")
    logger.info(f"Total steps: {result['step_count']}")
    logger.info(f"Papers retrieved: {len(result['papers'])}")
    logger.info(f"Summaries extracted: {len(result['paper_summaries'])}")
    logger.info("")
    logger.info("=== Messages from all nodes ===")
    for msg in result["messages"]:
        logger.info(f"  {msg}")
    
    # 5. 保存报告
    Path("output").mkdir(exist_ok=True)
    output_path = Path("output") / "report_3.md"
    output_path.write_text(result["final_report"], encoding="utf-8")
    logger.info(f"\n✅ Report saved to: {output_path}")


if __name__ == "__main__":
    main()