"""
Planner Agent (真实版) - 用 GLM 把用户问题拆解成子任务。

设计要点:
1. Prompt 要求 LLM 输出 JSON,便于程序解析
2. 包含 robust 的 JSON 解析(LLM 可能输出 markdown 包裹的 JSON)
3. 失败时降级到 fallback,不让 graph 崩溃
"""

import json
import logging
import re
from typing import List

from src.state import ResearchState, SubTask
from src.llm import get_llm

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """你是一个学术研究规划专家。你的任务是把用户的研究问题拆解成清晰、可执行的子任务。

输出要求:
1. 必须输出合法的 JSON 对象(不是数组),不要有任何额外文字
2. 包含两个字段:search_query_en 和 sub_tasks
3. search_query_en:把用户问题转化为适合 arXiv 检索的英文关键词(5-10 个词)
4. sub_tasks:子任务数组,每项包含 task_id、description、status("pending")
5. 子任务数量控制在 3-6 个
"""

PLANNER_USER_PROMPT_TEMPLATE = """研究问题:{query}

请输出 JSON 对象,格式如下:
{{
  "search_query_en": "english keywords for arxiv search",
  "sub_tasks": [
    {{"task_id": "1", "description": "...", "status": "pending"}},
    ...
  ]
}}

只输出 JSON,不要 markdown 代码块标记,不要解释。"""


PLANNER_USER_PROMPT_TEMPLATE = """研究问题:{query}

请输出 JSON 数组,格式如下:
[
  {{"task_id": "1", "description": "...", "status": "pending"}},
  ...
]

只输出 JSON,不要 markdown 代码块标记,不要解释。"""


def extract_json(text: str) -> str:
    """
    从 LLM 输出中提取 JSON。
    
    LLM 经常输出 ```json ... ``` 这样的格式,要清理掉。
    """
    text = text.strip()
    # 去掉 markdown 代码块标记
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_planner_output(llm_output: str):
    """解析 Planner 的完整输出(含 search_query_en + sub_tasks)"""
    try:
        cleaned = extract_json(llm_output)
        data = json.loads(cleaned)

        search_query_en = data.get("search_query_en", "")
        raw_tasks = data.get("sub_tasks", [])

        sub_tasks = [
            SubTask(
                task_id=str(item.get("task_id", i + 1)),
                description=item["description"],
                status="pending",
            )
            for i, item in enumerate(raw_tasks)
        ]
        return search_query_en, sub_tasks

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to parse planner output: {e}")
        return "", [
            SubTask(task_id="1", description="检索相关论文", status="pending"),
            SubTask(task_id="2", description="提取方法与数据", status="pending"),
            SubTask(task_id="3", description="生成综述", status="pending"),
        ]


def planner_node(state: ResearchState) -> dict:
    logger.info(f"[Planner] Planning for: {state['query']}")
    llm = get_llm()
    user_prompt = PLANNER_USER_PROMPT_TEMPLATE.format(query=state["query"])

    try:
        response = llm.chat(
            prompt=user_prompt,
            system=PLANNER_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=4096,
        )
        search_query_en, sub_tasks = parse_planner_output(response)
        logger.info(f"[Planner] LLM raw response length: {len(response)} chars")
        logger.info(f"[Planner] LLM response preview: {response[:200]!r}")
        logger.info(f"[Planner] English query: '{search_query_en}'")
        logger.info(f"[Planner] Sub-tasks: {len(sub_tasks)}")

        return {
            "sub_tasks": sub_tasks,
            "search_query_en": search_query_en,
            "messages": [
                f"[Planner] Created {len(sub_tasks)} sub-tasks, "
                f"search_query_en='{search_query_en}'"
            ],
            "step_count": state["step_count"] + 1,
        }

    except Exception as e:
        logger.error(f"[Planner] Failed: {e}")
        return {
            "sub_tasks": [],
            "search_query_en": state["query"],  # fallback 用原始 query
            "error": str(e),
            "messages": [f"[Planner] FAILED: {e}"],
            "step_count": state["step_count"] + 1,
        }

