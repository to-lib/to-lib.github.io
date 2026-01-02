---
sidebar_position: 18
title: 🌊 流式处理
---

# 流式处理 (Streaming)

流式处理是让 LLM 逐步返回生成内容的技术，而不是等待完整响应。这能显著提升用户体验，让用户更快看到输出。

## 为什么需要流式处理？

| 场景           | 非流式         | 流式           |
| -------------- | -------------- | -------------- |
| 首字节时间     | 等待完整生成   | 几百毫秒       |
| 用户体验       | 长时间等待     | 实时反馈       |
| 长文本生成     | 可能超时       | 持续输出       |
| 资源利用       | 一次性加载     | 渐进式处理     |

## OpenAI 流式 API

### 基础用法

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "写一首关于春天的诗"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 异步流式

```python
from openai import AsyncOpenAI
import asyncio

client = AsyncOpenAI()

async def stream_chat(prompt: str):
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

asyncio.run(stream_chat("解释什么是机器学习"))
```

### 收集完整响应

```python
def stream_with_full_response(prompt: str) -> tuple[str, dict]:
    """流式输出同时收集完整响应"""
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True}  # 包含 token 使用量
    )
    
    full_content = ""
    usage = None
    
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
            print(content, end="", flush=True)
        
        # 最后一个 chunk 包含 usage
        if chunk.usage:
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens
            }
    
    print()  # 换行
    return full_content, usage

content, usage = stream_with_full_response("你好")
print(f"Token 使用: {usage}")
```

## Anthropic 流式 API

### 基础用法

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "写一个 Python 快速排序"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### 事件处理

```python
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好"}]
) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            print(event.delta.text, end="", flush=True)
        elif event.type == "message_stop":
            print("\n[完成]")
        elif event.type == "message_delta":
            print(f"\n[停止原因: {event.delta.stop_reason}]")
```

### 获取最终消息

```python
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    
    # 获取完整消息对象
    final_message = stream.get_final_message()
    print(f"\nToken 使用: {final_message.usage}")
```

## 流式 Function Calling

### OpenAI 流式工具调用

```python
import json

def stream_with_tools(prompt: str, tools: list):
    """流式处理带工具调用"""
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        stream=True
    )
    
    tool_calls = {}
    content = ""
    
    for chunk in stream:
        delta = chunk.choices[0].delta
        
        # 处理文本内容
        if delta.content:
            content += delta.content
            print(delta.content, end="", flush=True)
        
        # 处理工具调用
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc.id,
                        "name": tc.function.name if tc.function else "",
                        "arguments": ""
                    }
                if tc.function and tc.function.arguments:
                    tool_calls[idx]["arguments"] += tc.function.arguments
    
    return content, list(tool_calls.values())

# 使用
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]

content, tool_calls = stream_with_tools("北京天气怎么样？", tools)
if tool_calls:
    print(f"\n工具调用: {tool_calls}")
```

## Web 应用集成

### FastAPI SSE

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI()

async def generate_stream(prompt: str):
    """生成 SSE 流"""
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            # SSE 格式
            yield f"data: {json.dumps({'content': content})}\n\n"
    
    yield "data: [DONE]\n\n"

@app.get("/chat/stream")
async def chat_stream(prompt: str):
    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

### 前端 JavaScript 消费

```javascript
async function streamChat(prompt) {
    const response = await fetch(`/chat/stream?prompt=${encodeURIComponent(prompt)}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        const lines = text.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') {
                    console.log('Stream completed');
                    return;
                }
                const parsed = JSON.parse(data);
                // 更新 UI
                document.getElementById('output').textContent += parsed.content;
            }
        }
    }
}
```

### React Hook

```typescript
import { useState, useCallback } from 'react';

function useStreamChat() {
    const [content, setContent] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    
    const sendMessage = useCallback(async (prompt: string) => {
        setIsLoading(true);
        setContent('');
        
        try {
            const response = await fetch('/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            });
            
            const reader = response.body?.getReader();
            if (!reader) return;
            
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const text = decoder.decode(value);
                // 解析 SSE 数据
                const matches = text.matchAll(/data: ({.*?})\n/g);
                for (const match of matches) {
                    const data = JSON.parse(match[1]);
                    setContent(prev => prev + data.content);
                }
            }
        } finally {
            setIsLoading(false);
        }
    }, []);
    
    return { content, isLoading, sendMessage };
}
```

## LangChain 流式处理

### 基础流式

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", streaming=True)

for chunk in llm.stream([HumanMessage(content="写一首诗")]):
    print(chunk.content, end="", flush=True)
```

### 流式 Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("用 {language} 解释 {topic}")
chain = prompt | llm | StrOutputParser()

for chunk in chain.stream({"language": "简单的语言", "topic": "量子计算"}):
    print(chunk, end="", flush=True)
```

### 异步流式

```python
async def async_stream():
    async for chunk in llm.astream([HumanMessage(content="你好")]):
        print(chunk.content, end="", flush=True)

import asyncio
asyncio.run(async_stream())
```

### 流式事件

```python
async def stream_events():
    """获取详细的流式事件"""
    chain = prompt | llm | StrOutputParser()
    
    async for event in chain.astream_events(
        {"language": "中文", "topic": "AI"},
        version="v2"
    ):
        kind = event["event"]
        
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        elif kind == "on_chain_end":
            print("\n[Chain 完成]")
```

## 流式处理最佳实践

### 1. 超时处理

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def stream_with_timeout(prompt: str, timeout: float = 30.0):
    """带超时的流式处理"""
    try:
        stream = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            ),
            timeout=timeout
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except asyncio.TimeoutError:
        yield "\n[超时]"
```

### 2. 错误恢复

```python
async def resilient_stream(prompt: str, max_retries: int = 3):
    """带重试的流式处理"""
    for attempt in range(max_retries):
        try:
            stream = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return
            
        except Exception as e:
            if attempt == max_retries - 1:
                yield f"\n[错误: {e}]"
            else:
                await asyncio.sleep(2 ** attempt)
```

### 3. 取消处理

```python
import asyncio

async def cancellable_stream(prompt: str):
    """可取消的流式处理"""
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    try:
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except asyncio.CancelledError:
        print("\n[已取消]")
        raise

# 使用
async def main():
    task = asyncio.create_task(consume_stream())
    await asyncio.sleep(2)
    task.cancel()  # 2秒后取消
```

### 4. 缓冲处理

```python
async def buffered_stream(prompt: str, buffer_size: int = 10):
    """缓冲流式输出，减少 UI 更新频率"""
    buffer = ""
    
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            buffer += chunk.choices[0].delta.content
            
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = ""
    
    if buffer:
        yield buffer
```

## 性能优化

### 1. 连接复用

```python
from openai import OpenAI
import httpx

# 使用自定义 HTTP 客户端
http_client = httpx.Client(
    limits=httpx.Limits(max_keepalive_connections=10),
    timeout=httpx.Timeout(60.0, connect=5.0)
)

client = OpenAI(http_client=http_client)
```

### 2. 并发流式请求

```python
async def parallel_streams(prompts: list[str]):
    """并发处理多个流式请求"""
    async def process_one(prompt: str, index: int):
        result = ""
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
        return index, result
    
    tasks = [process_one(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return dict(results)
```

## 延伸阅读

- [OpenAI Streaming](https://platform.openai.com/docs/api-reference/streaming)
- [Anthropic Streaming](https://docs.anthropic.com/claude/reference/messages-streaming)
- [LangChain Streaming](https://python.langchain.com/docs/expression_language/streaming)
