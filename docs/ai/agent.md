---
sidebar_position: 2
title: 🤖 AI Agent (智能体)
---

# AI Agent (智能体)

AI Agent（人工智能代理/智能体）是指能够感知环境、进行推理决策并采取行动以实现特定目标的系统。在 LLM 时代，Agent 通常指以大语言模型为核心大脑，具备规划、记忆和工具使用能力的系统。

## 核心架构 (CoG 架构)

一个典型的 AI Agent 通常包含以下核心组件：

### 1. 🧠 大脑 (Profile / Persona)

- **角色设定**：定义 Agent 的性格、职责和目标。
- **核心模型**：通常由 LLM (如 GPT-4, Claude 3, Gemini) 充当，负责推理和生成。

### 2. 📝 记忆 (Memory)

- **短期记忆**：上下文窗口中的对话历史。
- **长期记忆**：通过向量数据库 (Vector DB) 存储和检索的历史信息。

### 3. 📅 规划 (Planning)

- **任务分解**：将复杂目标分解为可执行的子任务 (如 Chain of Thought)。
- **反思与修正**：根据执行结果调整计划 (如 ReAct, Reflexion)。

### 4. 🛠️ 工具使用 (Action / Tools)

- **API 调用**：通过 Function Calling 访问外部 API (搜索、计算、数据库)。
- **物理操作**：如果是具身智能 (Embodied AI)，则涉及物理世界的操作。

## 常见 Agent 模式

### ReAct (Reasoning + Acting)

模型在执行行动前先进行思考 (Reasoning)，然后执行行动 (Acting)，并观察结果。这是一种最基础且有效的 Agent 模式。

### AutoGPT / BabyAGI

通过循环机制，让 Agent 自动生成任务列表、设定优先级并执行，直到达成最终目标。

### Multi-Agent (多智能体协作)

多个拥有不同角色和专长的 Agent 互相协作解决问题。例如：

- **MetaGPT**: 模拟软件开发团队 (产品经理、架构师、工程师) 协作写代码。
- **Microsoft AutoGen**: 灵活的多 Agent 对话框架。

## 应用场景

- **个人助理**：日程管理、邮件处理、信息检索。
- **客户服务**：智能客服，自动处理复杂的用户请求。
- **软件开发**：自动编写、测试和修复代码。
- **数据分析**：自动查询数据库、生成图表和分析报告。

## 挑战与未来

- **上下文限制**：虽然 LLM 的上下文越来越长，但仍有限制。
- **幻觉问题**：Agent 可能会生成错误的计划或调用不存在的工具。
- **循环陷阱**：Agent 可能陷入死循环无法跳出。
- **安全性**：自主行动可能带来的安全风险。

## 代码实现示例

### 使用 LangChain 构建 ReAct Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# 定义工具
@tool
def search(query: str) -> str:
    """搜索网络获取最新信息"""
    # 实际应用中调用搜索 API
    return f"搜索结果：关于 '{query}' 的信息..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

tools = [search, calculator]

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ReAct Prompt 模板
prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

# 创建 Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行
result = agent_executor.invoke({"input": "2024年诺贝尔物理学奖获得者是谁？"})
print(result["output"])
```

### 使用 OpenAI Function Calling 构建简单 Agent

```python
from openai import OpenAI
import json

client = OpenAI()

# 定义可用工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]

def get_weather(city: str) -> dict:
    """模拟天气查询"""
    return {"city": city, "weather": "晴", "temp": "22°C"}

def run_agent(user_input: str):
    messages = [{"role": "user", "content": user_input}]

    # Agent 循环
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # 检查是否需要调用工具
        if not message.tool_calls:
            return message.content

        messages.append(message)

        # 执行工具调用
        for tool_call in message.tool_calls:
            if tool_call.function.name == "get_weather":
                args = json.loads(tool_call.function.arguments)
                result = get_weather(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

# 使用
print(run_agent("北京今天天气怎么样？"))
```

## 框架与工具推荐

| 框架                          | 特点               | 适用场景        |
| ----------------------------- | ------------------ | --------------- |
| **LangChain**                 | 功能全面，生态丰富 | 通用 Agent 开发 |
| **LangGraph**                 | 支持复杂工作流     | 多步骤任务编排  |
| **AutoGPT**                   | 自主任务规划       | 研究探索        |
| **CrewAI**                    | 多 Agent 协作      | 团队模拟        |
| **Microsoft Semantic Kernel** | .NET/Python 支持   | 企业集成        |

## 延伸阅读

- [LangChain Agent 文档](https://python.langchain.com/docs/modules/agents/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
