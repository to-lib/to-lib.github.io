---
sidebar_position: 5
title: 🔧 Function Calling
---

# Function Calling (函数调用)

Function Calling 是让 LLM 能够调用外部函数或 API 的能力。模型不直接执行代码，而是输出结构化的函数调用请求，由应用程序执行后将结果返回给模型。

## 工作原理

```
用户输入 → LLM 分析 → 决定调用函数 → 输出函数调用 JSON
    ↓
应用程序执行函数 → 获取结果 → 返回给 LLM
    ↓
LLM 基于结果生成最终回答 → 返回用户
```

## OpenAI Function Calling

### 定义工具

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "搜索产品信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "最高价格"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "books"],
                        "description": "产品类别"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

### 完整调用流程

```python
from openai import OpenAI
import json

client = OpenAI()

def get_weather(city: str, unit: str = "celsius") -> dict:
    """模拟天气 API 调用"""
    # 实际应用中调用真实的天气 API
    return {
        "city": city,
        "temperature": 22,
        "unit": unit,
        "condition": "晴朗"
    }

def search_products(query: str, max_price: float = None, category: str = None) -> list:
    """模拟产品搜索"""
    return [
        {"name": f"{query} 产品 A", "price": 99.0},
        {"name": f"{query} 产品 B", "price": 199.0}
    ]

# 函数映射表
available_functions = {
    "get_weather": get_weather,
    "search_products": search_products
}

def run_conversation(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    # 第一次调用：让模型决定是否调用函数
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # 模型自动决定
    )

    response_message = response.choices[0].message

    # 检查是否需要调用函数
    if response_message.tool_calls:
        messages.append(response_message)

        # 执行所有函数调用
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # 调用函数
            function_response = available_functions[function_name](**function_args)

            # 将函数结果添加到消息中
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response, ensure_ascii=False)
            })

        # 第二次调用：让模型基于函数结果生成回答
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        return final_response.choices[0].message.content

    return response_message.content

# 使用示例
result = run_conversation("北京今天天气怎么样？")
print(result)
```

### tool_choice 选项

| 值                                                  | 说明                     |
| --------------------------------------------------- | ------------------------ |
| `"auto"`                                            | 模型自动决定是否调用函数 |
| `"none"`                                            | 禁止调用函数             |
| `"required"`                                        | 必须调用至少一个函数     |
| `{"type": "function", "function": {"name": "xxx"}}` | 强制调用指定函数         |

## Anthropic Tool Use

### 定义工具

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    }
]
```

### 调用流程

```python
def run_claude_conversation(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # 检查是否需要调用工具
    if response.stop_reason == "tool_use":
        # 找到工具调用块
        tool_use_block = next(
            block for block in response.content
            if block.type == "tool_use"
        )

        # 执行工具
        tool_result = get_weather(**tool_use_block.input)

        # 继续对话
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": json.dumps(tool_result)
            }]
        })

        # 获取最终回答
        final_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        return final_response.content[0].text

    return response.content[0].text
```

## 并行函数调用

OpenAI 支持在一次响应中返回多个函数调用：

```python
# 用户问题："北京和上海的天气分别怎么样？"
# 模型可能返回两个 tool_calls

for tool_call in response_message.tool_calls:
    # 可以并行执行这些调用
    pass
```

## 最佳实践

### 1. 函数描述要清晰

```python
# ✅ 好的描述
{
    "name": "send_email",
    "description": "发送电子邮件给指定收件人。支持 HTML 格式内容和附件。",
    "parameters": {
        "properties": {
            "to": {
                "type": "string",
                "description": "收件人邮箱地址，如 user@example.com"
            }
        }
    }
}

# ❌ 差的描述
{
    "name": "send_email",
    "description": "发邮件",
    "parameters": {
        "properties": {
            "to": {"type": "string"}
        }
    }
}
```

### 2. 参数使用 enum 约束

```python
"parameters": {
    "properties": {
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "任务优先级"
        }
    }
}
```

### 3. 错误处理

```python
def safe_function_call(func, args):
    try:
        result = func(**args)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 4. 函数数量控制

- 相关函数不超过 10-15 个
- 过多函数会降低模型选择准确性
- 考虑使用分层结构或动态加载

## 安全考虑

:::warning 安全提示

- **验证参数**：永远不要信任 LLM 生成的参数
- **权限控制**：限制函数的操作范围
- **审计日志**：记录所有函数调用
- **敏感操作确认**：危险操作需要人工确认
  :::

```python
DANGEROUS_FUNCTIONS = ["delete_file", "run_sql", "send_money"]

def execute_function(name, args):
    if name in DANGEROUS_FUNCTIONS:
        if not get_user_confirmation(name, args):
            return {"error": "用户取消操作"}

    return available_functions[name](**args)
```

## 与 MCP 的关系

Function Calling 是模型能力，MCP 是连接标准：

| 特性     | Function Calling | MCP          |
| -------- | ---------------- | ------------ |
| 定位     | 模型原生能力     | 连接协议标准 |
| 工具定义 | API 私有格式     | 统一标准格式 |
| 可移植性 | 绑定特定模型     | 跨平台通用   |
| 复杂度   | 较简单           | 功能更丰富   |

MCP Server 内部通常也会使用 Function Calling 来实现工具调用。
