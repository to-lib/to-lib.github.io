---
sidebar_position: 7
title: 🤖 AI Agent (智能体)
---

# AI Agent (智能体)

AI Agent（人工智能代理/智能体）是指能够感知环境、进行推理决策并采取行动以实现特定目标的系统。
在 LLM（大语言模型）时代，Agent 被定义为：**Agent = LLM + Planning + Memory + Tools**。
它不仅能够生成文本，还能主动“做事”，是通往 AGI（通用人工智能）的重要路径。

## 核心架构 (The Agent System)

一个典型的 AI Agent 系统架构如下图所示：

```mermaid
graph TD
    User([User]) <--> Interface
    subgraph Agent System
        Interface[Human-Agent Interface]
        brain[Brain (LLM)]
        memory[Memory]
        planning[Planning]
        tools[Tools / Actions]

        Interface --> brain
        brain <--> memory
        brain -- Thought/Plan --> planning
        planning -- Refinement --> brain
        brain -- Action Call --> tools
        tools -- Observation --> brain
    end
    tools <--> Environment((Environment/API))
```

### 1. 🧠 大脑 (Brain)

核心主要由 **LLM (Large Language Model)** 担当，如 GPT-4, Claude 3.5, Gemini 1.5 Pro 等。

- **职责**：自然语言理解、知识检索、推理规划、决策生成。
- **角色设定 (Persona)**：通过 Prompt Engineering 定义 Agent 的性格、专业领域和行为准则。

### 2. 📝 记忆 (Memory)

记忆模块使 Agent 能够从过去的经验中学习并保持上下文连续性。

- **感觉记忆 (Sensory Memory)**：作为输入的原始数据（文本、图像、音频等）。
- **短期记忆 (Short-term Memory)**：
  - 即模型的**上下文窗口 (Context Window)**。
  - 存储当前的对话历史、临时的思考过程。
  - 受限于 Context Length（如 128k, 1M token）。
- **长期记忆 (Long-term Memory)**：
  - 类似于人类在大脑中永久存储知识。
  - 通常使用 **向量数据库 (Vector Database)** 实现（如 Pinecone, Milvus, Weaviate）。
  - 能够进行快速检索 (Retrieval)，通过 RAG 技术增强 Agent 的知识库。

### 3. 📅 规划 (Planning)

Agent 在行动之前需要对复杂任务进行拆解和规划。

- **任务分解 (Decomposition)**：
  - 将宏大的目标拆解为更小、可执行的子步骤。
  - 技术手段：Chain of Thought (CoT), Tree of Thoughts (ToT)。
- **反思与修正 (Self-Reflection)**：
  - 对过去的行动和产生的结果进行自我批评和反思，从错误中修正。
  - 代表模式：**ReAct**, **Reflexion**。

### 4. 🛠️ 工具使用 (Tools / Action)

Agent 连接外部世界的桥梁，使其具备“手”和“脚”。

- **工具类型**：
  - **信息检索**：Google Search, Wikipedia, RAG Pipeline。
  - **代码执行**：Python REPL, Shell。
  - **API 调用**：发送邮件、操作日历、查询天气、调用企业内部 API。
- **实现方式**：
  - **Function Calling**: OpenAI 等模型原生支持的结构化输出，精准调用函数。
  - **JSON Mode**: 强制模型输出 JSON 格式来描述动作。

---

## 常见 Agent 设计模式 (Design Patterns)

### 1. ReAct (Reasoning + Acting)

最经典的单 Agent 模式。模型在执行行动前先进行显式的思考 (Thought)，然后执行行动 (Action)，并观察行动的结果 (Observation)。

- **流程**：`Thought -> Action -> Observation -> Thought -> ...`
- **优势**：通过“自言自语”的显式推理，减少了幻觉，增强了解决问题的鲁棒性。

### 2. Plan-and-Solve

先制定完整的计划，然后逐一执行，而不是像 ReAct 那样每一步都重新思考。适用于步骤清晰的任务。

### 3. Multi-Agent Collaboration (多智能体协作)

多个拥有不同角色（Profile）和专长的 Agent 互相协作解决问题。

- **垂直分工**：如软件开发流水线（产品经理 -> 架构师 -> 工程师 -> 测试）。
- **水平讨论**：多个 Agent 像圆桌会议一样讨论得出一个最佳方案。
- **代表框架**：**MetaGPT**, **AutoGen**, **CrewAI**。

### 4. Reflexion (反思)

在任务失败或结束后，Agent 对过程进行复盘，生成“经验教训”存入长期记忆，供下次任务参考。

---

## 应用场景

- **个人助理 (Copilot)**：日程管理、邮件处理、信息检索、文档摘要。
- **智能客服 (Customer Service)**：主动查询订单、处理退款、回答复杂业务问题。
- **软件工程 (Software Engineering)**：自动化代码编写、单元测试生成、Bug 修复 (如 Devin)。
- **数据分析 (Data Analyst)**：接收自然语言指令，自动编写 SQL/Python 查询数据库，生成图表报告。
- **科学研究 (Research)**：自动搜集论文、阅读摘要、生成综述。

---

## 挑战与局限

- **上下文限制 (Context Length)**：虽然窗口在变长，但无限长的记忆仍需依赖检索系统，会有精度损失。
- **幻觉 (Hallucination)**：Agent 可能会一本正经地胡说八道，或调用不存在的工具参数。
- **死循环 (Infinite Loops)**：Agent 可能陷入重复的思考或行动中无法跳出，需要设置最大迭代次数。
- **规划能力瓶颈**：面对极度复杂的长链路任务，LLM 可能会丢失目标或规划偏离。
- **安全性 (Safety)**：自主 Agent 可能执行危险操作（如删除文件、发送不当邮件），需要 **Human-in-the-loop** 机制。

---

## 代码实现示例

### 1. 使用 LangChain 构建 ReAct Agent

LangChain 是最流行的 Agent 开发框架之一，封装了大量 Tool 和 Agent 逻辑。

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# --- 1. 定义工具 ---
@tool
def search(query: str) -> str:
    """搜索网络获取最新信息"""
    # 实际应用中调用 Google/Bing Search API
    return f"搜索结果：关于 '{query}' 的最新信息..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

tools = [search, calculator]

# --- 2. 创建大脑 (LLM) ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- 3. 定义 ReAct Prompt ---
# 指导模型进行 Thought -> Action -> Observation 的循环
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

# --- 4. 初始化 Agent ---
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 5. 运行 ---
result = agent_executor.invoke({"input": "2024年诺贝尔物理学奖获得者是谁？他几岁了？"})
print(result["output"])
```

### 2. 使用 OpenAI 原生 Function Calling

不依赖第三方框架，直接使用 OpenAI API 构建轻量级 Agent。

```python
from openai import OpenAI
import json

client = OpenAI()

# 1. 定义工具描述 (JSON Schema)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如北京、上海"}
                },
                "required": ["city"]
            }
        }
    }
]

# 2. 工具的具体实现
def get_weather(city: str) -> dict:
    """模拟天气查询 API"""
    print(f">> 正在查询 {city} 的天气...")
    return {"city": city, "weather": "晴", "temp": "25°C"}

def run_agent(user_input: str):
    messages = [{"role": "user", "content": user_input}]

    # Agent 思考与执行循环
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto" # 让模型自动决定是否调用工具
        )

        message = response.choices[0].message

        # 如果模型返回了文本内容，直接返回给用户
        if message.tool_calls is None:
            return message.content

        # 如果模型决定调用工具
        messages.append(message) # 将助手的回复加入历史

        for tool_call in message.tool_calls:
            if tool_call.function.name == "get_weather":
                # 解析参数
                args = json.loads(tool_call.function.arguments)
                # 执行函数
                result = get_weather(**args)
                # 将结果反馈给模型
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

# 运行示例
print("Agent 回复:", run_agent("北京今天天气好吗？适合出去玩吗？"))
```

---

## 框架与工具生态

| 框架           | 类型      | 特点                                      | 适用场景                         |
| :------------- | :-------- | :---------------------------------------- | :------------------------------- |
| **LangChain**  | SDK       | 大而全，组件丰富，生态最强                | 通用 LLM 应用开发，快速原型      |
| **LangGraph**  | SDK       | 基于图（Graph）的编排，支持循环和状态管理 | 构建复杂的、有状态的多步 Agent   |
| **AutoGen**    | Framework | 微软出品，强大的多 Agent 对话框架         | 多角色协作，代码生成与执行       |
| **CrewAI**     | Framework | 专注于基于角色的多 Agent 编排             | 模拟团队工作流（如研究员+写手）  |
| **LlamaIndex** | SDK       | 专注数据索引与 RAG                        | 以数据为中心的 Agent，知识库问答 |

## 延伸阅读

- [Lil'Log: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) (Agent 领域的圣经)
- [LangChain Agents 文档](https://python.langchain.com/docs/modules/agents/)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
