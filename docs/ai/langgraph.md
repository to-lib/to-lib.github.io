---
sidebar_position: 16
title: 🔀 LangGraph 工作流编排
---

# LangGraph 工作流编排

LangGraph 是 LangChain 团队推出的工作流编排框架，专门用于构建复杂的、有状态的 AI Agent 应用。它基于图结构，支持循环、条件分支、人机协作等高级模式。

## 为什么需要 LangGraph？

传统的 LangChain Agent 是线性的，难以处理：

| 场景               | 传统 Agent | LangGraph |
| ------------------ | ---------- | --------- |
| 循环执行           | ❌         | ✅        |
| 条件分支           | 有限       | ✅        |
| 并行执行           | ❌         | ✅        |
| 人机协作（审批）   | ❌         | ✅        |
| 状态持久化         | ❌         | ✅        |
| 错误恢复           | 有限       | ✅        |

## 核心概念

```
┌─────────────────────────────────────────────────────────┐
│                      StateGraph                         │
│                                                         │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐          │
│   │  Node   │────▶│  Node   │────▶│  Node   │          │
│   │ (agent) │     │ (tools) │     │ (check) │          │
│   └─────────┘     └─────────┘     └────┬────┘          │
│        ▲                               │               │
│        │              ┌────────────────┴────────┐      │
│        │              ▼                         ▼      │
│        │         ┌─────────┐              ┌─────────┐  │
│        └─────────│ continue│              │   end   │  │
│                  └─────────┘              └─────────┘  │
└─────────────────────────────────────────────────────────┘
```

- **State（状态）**：在节点间传递的数据结构
- **Node（节点）**：执行具体逻辑的函数
- **Edge（边）**：节点之间的连接，可以是条件边
- **Graph（图）**：由节点和边组成的工作流

## 安装

```bash
pip install langgraph langchain-openai
```

## 基础示例：ReAct Agent

```python
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 2. 定义工具
@tool
def search(query: str) -> str:
    """搜索网络获取信息"""
    return f"搜索结果：关于 '{query}' 的信息..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

tools = [search, calculator]

# 3. 创建模型
model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

# 4. 定义节点函数
def agent_node(state: AgentState) -> dict:
    """Agent 决策节点"""
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """判断是否继续执行"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 5. 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# 设置入口
workflow.set_entry_point("agent")

# 添加条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# 工具执行后返回 agent
workflow.add_edge("tools", "agent")

# 编译图
app = workflow.compile()

# 6. 运行
result = app.invoke({
    "messages": [HumanMessage(content="北京今天天气怎么样？")]
})

for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")
```

## 高级模式

### 1. 并行执行

```python
from langgraph.graph import StateGraph
from typing import TypedDict
import asyncio

class ParallelState(TypedDict):
    query: str
    search_result: str
    analysis_result: str
    final_result: str

async def search_node(state: ParallelState) -> dict:
    """搜索节点"""
    # 模拟搜索
    await asyncio.sleep(1)
    return {"search_result": f"搜索结果: {state['query']}"}

async def analysis_node(state: ParallelState) -> dict:
    """分析节点"""
    # 模拟分析
    await asyncio.sleep(1)
    return {"analysis_result": f"分析结果: {state['query']}"}

def combine_node(state: ParallelState) -> dict:
    """合并结果"""
    return {
        "final_result": f"{state['search_result']} + {state['analysis_result']}"
    }

# 构建并行图
workflow = StateGraph(ParallelState)

workflow.add_node("search", search_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("combine", combine_node)

workflow.set_entry_point("search")

# 并行边：search 和 analysis 同时执行
workflow.add_edge("search", "combine")
workflow.add_edge("analysis", "combine")

# 需要使用 fan-out/fan-in 模式
# 实际上 LangGraph 会自动处理并行
```

### 2. 人机协作（Human-in-the-Loop）

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class ApprovalState(TypedDict):
    task: str
    plan: str
    approved: bool
    result: str

def plan_node(state: ApprovalState) -> dict:
    """生成计划"""
    plan = f"执行任务 '{state['task']}' 的计划：\n1. 步骤一\n2. 步骤二"
    return {"plan": plan}

def execute_node(state: ApprovalState) -> dict:
    """执行任务"""
    return {"result": f"任务完成: {state['task']}"}

def check_approval(state: ApprovalState) -> Literal["execute", "end"]:
    """检查是否批准"""
    if state.get("approved"):
        return "execute"
    return "end"

# 构建图
workflow = StateGraph(ApprovalState)

workflow.add_node("plan", plan_node)
workflow.add_node("execute", execute_node)

workflow.set_entry_point("plan")

workflow.add_conditional_edges(
    "plan",
    check_approval,
    {
        "execute": "execute",
        "end": END
    }
)

workflow.add_edge("execute", END)

# 使用检查点保存状态
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["execute"])

# 运行到需要审批的节点
config = {"configurable": {"thread_id": "1"}}
result = app.invoke({"task": "发送重要邮件", "approved": False}, config)

print("计划:", result["plan"])
print("等待人工审批...")

# 人工审批后继续执行
app.update_state(config, {"approved": True})
final_result = app.invoke(None, config)
print("结果:", final_result["result"])
```

### 3. 子图（Subgraph）

```python
from langgraph.graph import StateGraph, END

# 定义子图
def create_research_subgraph():
    """创建研究子图"""
    
    class ResearchState(TypedDict):
        topic: str
        sources: list
        summary: str
    
    def search_sources(state):
        return {"sources": [f"来源1: {state['topic']}", f"来源2: {state['topic']}"]}
    
    def summarize(state):
        return {"summary": f"关于 {state['topic']} 的总结: {state['sources']}"}
    
    subgraph = StateGraph(ResearchState)
    subgraph.add_node("search", search_sources)
    subgraph.add_node("summarize", summarize)
    subgraph.set_entry_point("search")
    subgraph.add_edge("search", "summarize")
    subgraph.add_edge("summarize", END)
    
    return subgraph.compile()

# 在主图中使用子图
class MainState(TypedDict):
    query: str
    research_result: str
    final_answer: str

research_app = create_research_subgraph()

def research_node(state: MainState) -> dict:
    """调用研究子图"""
    result = research_app.invoke({"topic": state["query"]})
    return {"research_result": result["summary"]}

def answer_node(state: MainState) -> dict:
    """生成最终答案"""
    return {"final_answer": f"基于研究: {state['research_result']}"}

main_workflow = StateGraph(MainState)
main_workflow.add_node("research", research_node)
main_workflow.add_node("answer", answer_node)
main_workflow.set_entry_point("research")
main_workflow.add_edge("research", "answer")
main_workflow.add_edge("answer", END)

main_app = main_workflow.compile()
```

### 4. 错误处理与重试

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class RetryState(TypedDict):
    input: str
    result: str
    error: str
    retry_count: int

def process_node(state: RetryState) -> dict:
    """处理节点（可能失败）"""
    try:
        # 模拟可能失败的操作
        if state.get("retry_count", 0) < 2:
            raise Exception("临时错误")
        return {"result": f"处理成功: {state['input']}", "error": ""}
    except Exception as e:
        return {"error": str(e), "retry_count": state.get("retry_count", 0) + 1}

def should_retry(state: RetryState) -> str:
    """判断是否重试"""
    if state.get("error") and state.get("retry_count", 0) < 3:
        return "retry"
    elif state.get("error"):
        return "fail"
    return "success"

workflow = StateGraph(RetryState)

workflow.add_node("process", process_node)
workflow.add_node("handle_error", lambda s: {"result": f"最终失败: {s['error']}"})

workflow.set_entry_point("process")

workflow.add_conditional_edges(
    "process",
    should_retry,
    {
        "retry": "process",  # 重试
        "fail": "handle_error",
        "success": END
    }
)

workflow.add_edge("handle_error", END)

app = workflow.compile()
```

## 状态持久化

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# SQLite 持久化
with SqliteSaver.from_conn_string(":memory:") as memory:
    app = workflow.compile(checkpointer=memory)
    
    # 运行并保存状态
    config = {"configurable": {"thread_id": "user-123"}}
    result = app.invoke({"messages": [HumanMessage("你好")]}, config)
    
    # 后续可以恢复状态继续对话
    result = app.invoke({"messages": [HumanMessage("继续上次的话题")]}, config)

# PostgreSQL 持久化（生产环境）
# with PostgresSaver.from_conn_string("postgresql://...") as memory:
#     app = workflow.compile(checkpointer=memory)
```

## 可视化

```python
# 生成 Mermaid 图
print(app.get_graph().draw_mermaid())

# 生成 PNG 图片（需要安装 graphviz）
# app.get_graph().draw_png("workflow.png")
```

## 实战：多步骤文档处理

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class DocProcessState(TypedDict):
    documents: List[str]
    chunks: List[str]
    embeddings: List[List[float]]
    index_status: str

def load_documents(state: DocProcessState) -> dict:
    """加载文档"""
    # 模拟加载
    return {"documents": ["文档1内容", "文档2内容"]}

def chunk_documents(state: DocProcessState) -> dict:
    """切分文档"""
    chunks = []
    for doc in state["documents"]:
        chunks.extend([f"{doc}_chunk1", f"{doc}_chunk2"])
    return {"chunks": chunks}

def embed_chunks(state: DocProcessState) -> dict:
    """生成向量"""
    embeddings = [[0.1] * 1536 for _ in state["chunks"]]
    return {"embeddings": embeddings}

def index_vectors(state: DocProcessState) -> dict:
    """索引向量"""
    return {"index_status": f"已索引 {len(state['embeddings'])} 个向量"}

# 构建流水线
workflow = StateGraph(DocProcessState)

workflow.add_node("load", load_documents)
workflow.add_node("chunk", chunk_documents)
workflow.add_node("embed", embed_chunks)
workflow.add_node("index", index_vectors)

workflow.set_entry_point("load")
workflow.add_edge("load", "chunk")
workflow.add_edge("chunk", "embed")
workflow.add_edge("embed", "index")
workflow.add_edge("index", END)

app = workflow.compile()

result = app.invoke({})
print(result["index_status"])
```

## 最佳实践

1. **状态设计**：保持状态简洁，只存储必要数据
2. **节点粒度**：每个节点做一件事，便于调试和复用
3. **错误处理**：为关键节点添加重试和降级逻辑
4. **持久化**：生产环境使用数据库持久化状态
5. **可观测性**：添加日志和追踪，便于排查问题

## 延伸阅读

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph 示例](https://github.com/langchain-ai/langgraph/tree/main/examples)
