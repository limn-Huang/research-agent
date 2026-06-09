"""
诊断脚本:验证 GLM 思考模型的 max_tokens 双重语义问题。

背景:GLM-5.1 是强制思考模型,max_tokens 参数同时控制"思考 tokens + 输出 tokens"。
当 max_tokens 设得太小(<1024),思考过程吃光预算,LLM 返回空字符串而不报错。

这是项目踩坑后沉淀的诊断脚本,可用于评估任何"思考模型"的 max_tokens 安全值。
"""


"""快速诊断:GLM-5.1 是否能输出 JSON"""
from dotenv import load_dotenv
load_dotenv()

from src.llm import get_llm

llm = get_llm()

# 测试 1:简单 JSON 要求
print("=" * 60)
print("Test 1: 简单 JSON")
print("=" * 60)
r1 = llm.chat(
    prompt='输出一个 JSON 对象,包含字段 name 和 age,name="张三",age=25。',
    temperature=0.0,
    max_tokens=100,
)
print(f"Length: {len(r1)}, Content: {r1!r}")

# 测试 2:严格 JSON(模拟你 prompt 风格)
print("\n" + "=" * 60)
print("Test 2: 严格 JSON 风格(模拟 Planner)")
print("=" * 60)
r2 = llm.chat(
    prompt="""请输出 JSON:
{"key": "value"}

只输出 JSON,不要 markdown 代码块标记,不要解释。""",
    system="你必须输出合法的 JSON,不要有任何额外文字。",
    temperature=0.0,
    max_tokens=100,
)
print(f"Length: {len(r2)}, Content: {r2!r}")

# 测试 3:用 markdown 包裹的请求
print("\n" + "=" * 60)
print("Test 3: 允许 markdown 包裹")
print("=" * 60)
r3 = llm.chat(
    prompt='请用 JSON 格式回答,name=张三, age=25。可以用 markdown 代码块。',
    temperature=0.0,
    max_tokens=100,
)
print(f"Length: {len(r3)}, Content: {r3!r}")