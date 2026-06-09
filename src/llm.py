"""
GLM 客户端封装。

为什么要单独封装而不是直接调 openai SDK:
1. 集中管理 base_url / api_key / model 配置(改一处全局生效)
2. 统一加重试、日志、错误处理
3. 后续切换模型(GLM → GPT → Claude)只改这一处
4. 方便加 mock(测试时不真调 API)
"""

import os
import logging
from typing import Optional, List, Dict, Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GLMClient:
    """
    GLM 客户端 (通过 OpenAI 兼容接口)
    
    用法:
        client = GLMClient()
        response = client.chat("你好")
        # 或带历史
        response = client.chat_with_messages([
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"}
        ])
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        self.model = model or os.getenv("GLM_MAIN_MODEL", "glm-5.1")
        self.base_url = base_url or os.getenv(
            "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
        )
        
        if not self.api_key:
            raise ValueError("ZHIPUAI_API_KEY not set. Check your .env file.")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"GLMClient initialized: model={self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """
        单轮对话,适合大多数 agent 节点。
        
        Args:
            prompt: 用户输入
            system: 系统提示(可选)
            temperature: 采样温度,0=确定性,1=多样性
            max_tokens: 最大输出 token
        
        Returns:
            模型生成的字符串
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_with_messages(messages, temperature, max_tokens)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def chat_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """
        多轮对话,直接传 messages list。
        
        装饰器 @retry:
        - 失败自动重试 3 次
        - 指数退避:2s → 4s → 8s,最大 10s
        - reraise=True 表示重试用尽后抛出原始异常
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            logger.debug(f"LLM response: {content[:200]}...")
            return content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise


# 全局单例(避免每个节点都创建一个 client)
_default_client: Optional[GLMClient] = None


def get_llm() -> GLMClient:
    """获取默认 LLM 客户端(单例模式)"""
    global _default_client
    if _default_client is None:
        _default_client = GLMClient()
    return _default_client


if __name__ == "__main__":
    # 单元测试:运行 `python src/llm.py` 测试是否能调通
    logging.basicConfig(level=logging.INFO)
    
    print("Testing GLM client...")
    llm = get_llm()
    response = llm.chat(
        prompt="用一句话介绍 LangGraph 是什么。",
        temperature=0.3,
    )
    print(f"Response: {response}")
    print("\n✅ GLM client works!" if response else "❌ Empty response")