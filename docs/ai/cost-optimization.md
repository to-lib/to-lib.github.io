---
sidebar_position: 27
title: 💰 成本优化
---

# 成本优化

AI 应用的成本主要来自 API 调用费用。本文介绍各种降低成本的策略。

## 成本构成

```
总成本 = 输入 Token × 输入价格 + 输出 Token × 输出价格
```

### 主流模型价格（2025）

| 模型 | 输入价格 | 输出价格 | 特点 |
|------|---------|---------|------|
| GPT-4o | $2.50/1M | $10.00/1M | 性能最强 |
| GPT-4o-mini | $0.15/1M | $0.60/1M | 性价比高 |
| Claude 3.5 Sonnet | $3.00/1M | $15.00/1M | 长上下文 |
| Claude 3.5 Haiku | $0.80/1M | $4.00/1M | 快速响应 |
| Gemini 1.5 Pro | $1.25/1M | $5.00/1M | 超长上下文 |
| Gemini 1.5 Flash | $0.075/1M | $0.30/1M | 极致性价比 |

## 策略 1: 模型选择

### 按任务选择模型

```python
from openai import OpenAI

client = OpenAI()

def select_model(task_type: str) -> str:
    """根据任务类型选择模型"""
    model_map = {
        "simple_qa": "gpt-4o-mini",      # 简单问答
        "classification": "gpt-4o-mini",  # 分类任务
        "summarization": "gpt-4o-mini",   # 摘要
        "code_generation": "gpt-4o",      # 代码生成
        "complex_reasoning": "o1-mini",   # 复杂推理
        "creative_writing": "gpt-4o",     # 创意写作
    }
    return model_map.get(task_type, "gpt-4o-mini")


def smart_query(query: str, task_type: str) -> str:
    model = select_model(task_type)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content
```

### 级联调用

```python
def cascade_query(query: str) -> str:
    """先用小模型，不确定时用大模型"""
    # 第一次尝试：小模型
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "回答问题。如果不确定，回复 'UNCERTAIN'。"
            },
            {"role": "user", "content": query}
        ]
    )
    
    answer = response.choices[0].message.content
    
    # 如果不确定，升级到大模型
    if "UNCERTAIN" in answer:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )
        answer = response.choices[0].message.content
    
    return answer
```

## 策略 2: Token 优化

### 精简 Prompt

```python
# ❌ 冗长的 Prompt
bad_prompt = """
你是一个非常专业的、经验丰富的、知识渊博的助手。
你需要仔细地、认真地、全面地回答用户的问题。
请确保你的回答是准确的、有帮助的、详细的。
用户的问题是：{question}
"""

# ✅ 精简的 Prompt
good_prompt = """
简洁回答：{question}
"""
```

### 压缩上下文

```python
def compress_context(context: str, max_tokens: int = 2000) -> str:
    """压缩上下文"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"将以下内容压缩到 {max_tokens} tokens 以内，保留关键信息。"
            },
            {"role": "user", "content": context}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
```

### 限制输出长度

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": query}],
    max_tokens=500  # 限制输出长度
)
```

## 策略 3: 缓存

### 语义缓存

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import hashlib

class SemanticCache:
    """语义缓存"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        self.cache = {}  # query_id -> response
        self.threshold = similarity_threshold
    
    def _get_similar(self, query: str) -> str | None:
        """查找相似查询"""
        results = self.vectorstore.similarity_search_with_score(query, k=1)
        
        if results and results[0][1] >= self.threshold:
            query_id = results[0][0].metadata.get("query_id")
            return self.cache.get(query_id)
        
        return None
    
    def get(self, query: str) -> str | None:
        return self._get_similar(query)
    
    def set(self, query: str, response: str):
        query_id = hashlib.md5(query.encode()).hexdigest()
        self.cache[query_id] = response
        self.vectorstore.add_texts(
            texts=[query],
            metadatas=[{"query_id": query_id}]
        )
    
    def query(self, query: str) -> str:
        # 检查缓存
        cached = self.get(query)
        if cached:
            print("Cache hit!")
            return cached
        
        # 调用 API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )
        
        result = response.choices[0].message.content
        self.set(query, result)
        return result
```

### Prompt Caching

参考 [Prompt Caching](./prompt-caching) 文档。

## 策略 4: 批处理

### 批量请求

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def batch_process(queries: list[str], batch_size: int = 10) -> list[str]:
    """批量处理请求"""
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        
        tasks = [
            async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": q}]
            )
            for q in batch
        ]
        
        responses = await asyncio.gather(*tasks)
        results.extend([r.choices[0].message.content for r in responses])
    
    return results
```

### OpenAI Batch API

```python
import json

def create_batch_file(requests: list[dict], filename: str):
    """创建批处理文件"""
    with open(filename, 'w') as f:
        for i, req in enumerate(requests):
            line = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": req["messages"]
                }
            }
            f.write(json.dumps(line) + "\n")

# 上传文件
batch_file = client.files.create(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch"
)

# 创建批处理任务
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"  # 24 小时内完成，价格减半
)

# 检查状态
status = client.batches.retrieve(batch.id)
print(f"Status: {status.status}")
```


## 策略 5: 本地模型

对于高频、低复杂度任务，使用本地模型。

```python
import ollama

def local_or_cloud(query: str, complexity: str = "low") -> str:
    """根据复杂度选择本地或云端模型"""
    if complexity == "low":
        # 使用本地模型
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": query}]
        )
        return response["message"]["content"]
    else:
        # 使用云端模型
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content
```

## 策略 6: 成本监控

### Token 计数

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """计算 token 数量"""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_cost(
    input_text: str,
    output_tokens: int,
    model: str = "gpt-4o"
) -> float:
    """估算成本"""
    prices = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    
    input_tokens = count_tokens(input_text, model)
    price = prices.get(model, prices["gpt-4o"])
    
    cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    return cost
```

### 使用量追踪

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class UsageTracker:
    """使用量追踪"""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    requests: int = 0
    daily_stats: dict = field(default_factory=dict)
    
    def record(self, input_tokens: int, output_tokens: int, model: str):
        self.requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # 计算成本
        prices = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        }
        price = prices.get(model, prices["gpt-4o"])
        cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
        self.total_cost += cost
        
        # 按日统计
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_stats:
            self.daily_stats[today] = {"tokens": 0, "cost": 0}
        self.daily_stats[today]["tokens"] += input_tokens + output_tokens
        self.daily_stats[today]["cost"] += cost
    
    def report(self) -> str:
        return f"""
使用报告：
- 总请求数：{self.requests}
- 总输入 tokens：{self.total_input_tokens:,}
- 总输出 tokens：{self.total_output_tokens:,}
- 总成本：${self.total_cost:.4f}
"""

tracker = UsageTracker()
```

## 成本优化清单

| 策略 | 节省比例 | 适用场景 |
|------|---------|---------|
| 使用小模型 | 90%+ | 简单任务 |
| Prompt Caching | 50-90% | 重复前缀 |
| 批处理 API | 50% | 非实时任务 |
| 语义缓存 | 变化大 | 重复查询 |
| 本地模型 | 100% | 高频低复杂度 |
| 压缩上下文 | 30-50% | 长文档 |
| 限制输出 | 20-40% | 简洁回答 |

## 最佳实践

1. **监控先行**：先了解成本分布再优化
2. **分级处理**：不同任务用不同模型
3. **缓存优先**：相似查询复用结果
4. **批量处理**：非实时任务用 Batch API
5. **设置预算**：配置用量告警

## 延伸阅读

- [OpenAI Pricing](https://openai.com/pricing)
- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)
- [tiktoken](https://github.com/openai/tiktoken)