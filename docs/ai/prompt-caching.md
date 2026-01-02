---
sidebar_position: 22
title: 💾 Prompt Caching
---

# Prompt Caching（提示缓存）

Prompt Caching 是 OpenAI 和 Anthropic 提供的功能，可以缓存重复使用的提示前缀，显著降低成本和延迟。

## 为什么需要 Prompt Caching？

| 场景               | 问题                     | Prompt Caching 效果 |
| ------------------ | ------------------------ | ------------------- |
| 长系统提示         | 每次都要处理相同内容     | 缓存后只处理一次    |
| RAG 固定上下文     | 相同文档重复发送         | 缓存文档内容        |
| 多轮对话           | 历史消息重复处理         | 缓存历史部分        |
| Few-shot 示例      | 相同示例重复发送         | 缓存示例            |

## 成本节省

### OpenAI

| 模型    | 正常输入价格 | 缓存输入价格 | 节省比例 |
| ------- | ------------ | ------------ | -------- |
| GPT-4o  | $2.50/1M     | $1.25/1M     | 50%      |
| o1      | $15.00/1M    | $7.50/1M     | 50%      |

### Anthropic

| 模型              | 正常输入价格 | 缓存写入价格 | 缓存读取价格 | 节省比例 |
| ----------------- | ------------ | ------------ | ------------ | -------- |
| Claude 3.5 Sonnet | $3.00/1M     | $3.75/1M     | $0.30/1M     | 90%      |
| Claude 3.5 Haiku  | $0.80/1M     | $1.00/1M     | $0.08/1M     | 90%      |

## OpenAI Prompt Caching

OpenAI 的 Prompt Caching 是**自动**的，无需额外配置。

### 工作原理

```
请求 1: [系统提示 2000 tokens] + [用户消息 100 tokens]
        ↓ 自动缓存前缀
请求 2: [系统提示 2000 tokens] + [用户消息 200 tokens]
        ↓ 命中缓存，只处理新增部分
```

### 缓存条件

- 提示前缀必须**完全相同**（逐字符匹配）
- 最小缓存长度：1024 tokens
- 缓存有效期：5-10 分钟（低流量时更短）
- 相同组织内的请求共享缓存

### 查看缓存命中

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一个专业的助手..." * 100},  # 长系统提示
        {"role": "user", "content": "你好"}
    ]
)

# 查看缓存信息
usage = response.usage
print(f"总输入 tokens: {usage.prompt_tokens}")
print(f"缓存命中 tokens: {usage.prompt_tokens_details.cached_tokens}")
```

### 优化缓存命中率

```python
# ✅ 好的做法：固定前缀，变化部分放后面
messages = [
    {"role": "system", "content": FIXED_SYSTEM_PROMPT},  # 固定
    {"role": "user", "content": FIXED_EXAMPLES},         # 固定示例
    {"role": "user", "content": user_input}              # 变化部分
]

# ❌ 不好的做法：变化部分在前面
messages = [
    {"role": "system", "content": f"当前时间：{datetime.now()}"},  # 每次都变
    {"role": "system", "content": FIXED_SYSTEM_PROMPT},
    {"role": "user", "content": user_input}
]
```

### 批量请求优化

```python
# 相同前缀的请求会共享缓存
async def batch_with_cache(prompts: list[str], system_prompt: str):
    """批量请求，共享系统提示缓存"""
    tasks = []
    for prompt in prompts:
        task = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        tasks.append(task)
    
    return await asyncio.gather(*tasks)
```

## Anthropic Prompt Caching

Anthropic 的 Prompt Caching 需要**显式标记**缓存断点。

### 基础用法

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "你是一个专业的助手，以下是你需要参考的文档：\n" + long_document,
            "cache_control": {"type": "ephemeral"}  # 标记缓存点
        }
    ],
    messages=[
        {"role": "user", "content": "总结文档的主要内容"}
    ]
)

# 查看缓存信息
print(f"输入 tokens: {response.usage.input_tokens}")
print(f"缓存创建 tokens: {response.usage.cache_creation_input_tokens}")
print(f"缓存读取 tokens: {response.usage.cache_read_input_tokens}")
```

### 多个缓存断点

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "你是一个代码助手。",
        },
        {
            "type": "text",
            "text": code_documentation,  # 代码文档
            "cache_control": {"type": "ephemeral"}  # 缓存点 1
        }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": few_shot_examples,  # Few-shot 示例
                    "cache_control": {"type": "ephemeral"}  # 缓存点 2
                },
                {
                    "type": "text",
                    "text": "请帮我写一个排序函数"
                }
            ]
        }
    ]
)
```

### 缓存条件

- 最小缓存长度：
  - Claude 3.5 Sonnet/Opus: 1024 tokens
  - Claude 3.5 Haiku: 2048 tokens
- 最多 4 个缓存断点
- 缓存有效期：5 分钟
- 必须使用 `cache_control` 显式标记

### 工具定义缓存

```python
tools = [
    {
        "name": "search",
        "description": "搜索文档",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
    },
    # ... 更多工具
]

# 缓存工具定义
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    extra_headers={
        "anthropic-beta": "prompt-caching-2024-07-31"
    },
    system=[
        {
            "type": "text",
            "text": "你是一个助手。",
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": "搜索相关文档"}]
)
```

## 实战应用

### 1. RAG 场景优化

```python
class CachedRAG:
    """带缓存的 RAG 系统"""
    
    def __init__(self, documents: str):
        self.documents = documents
        self.client = anthropic.Anthropic()
    
    def query(self, question: str) -> str:
        """查询时复用文档缓存"""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": f"你是一个问答助手。请根据以下文档回答问题：\n\n{self.documents}",
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response.content[0].text

# 使用
rag = CachedRAG(long_document)
# 第一次请求：创建缓存
answer1 = rag.query("文档的主题是什么？")
# 后续请求：命中缓存，成本降低 90%
answer2 = rag.query("作者是谁？")
answer3 = rag.query("主要结论是什么？")
```

### 2. 多轮对话优化

```python
class CachedChat:
    """带缓存的多轮对话"""
    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages = []
        self.client = anthropic.Anthropic()
    
    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        
        # 构建带缓存的消息
        cached_messages = []
        for i, msg in enumerate(self.messages[:-1]):  # 历史消息
            if i == len(self.messages) - 2:  # 最后一条历史消息加缓存
                cached_messages.append({
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                })
            else:
                cached_messages.append(msg)
        
        # 添加当前消息
        cached_messages.append(self.messages[-1])
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=cached_messages
        )
        
        assistant_message = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
```

### 3. Few-shot 学习优化

```python
def few_shot_with_cache(examples: str, query: str) -> str:
    """缓存 Few-shot 示例"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"以下是一些示例：\n\n{examples}",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": f"\n\n现在请处理：{query}"
                    }
                ]
            }
        ]
    )
    return response.content[0].text

# 示例
examples = """
输入：今天天气真好
输出：positive

输入：这个产品太差了
输出：negative

输入：还行吧
输出：neutral
"""

# 多次调用，示例部分被缓存
result1 = few_shot_with_cache(examples, "我很喜欢这个")
result2 = few_shot_with_cache(examples, "太糟糕了")
result3 = few_shot_with_cache(examples, "一般般")
```

## 监控与优化

### 缓存命中率监控

```python
class CacheMonitor:
    """缓存监控"""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.cached_tokens = 0
        self.requests = 0
    
    def record(self, usage):
        self.requests += 1
        self.total_input_tokens += usage.input_tokens
        
        # Anthropic
        if hasattr(usage, 'cache_read_input_tokens'):
            self.cached_tokens += usage.cache_read_input_tokens
        # OpenAI
        elif hasattr(usage, 'prompt_tokens_details'):
            self.cached_tokens += usage.prompt_tokens_details.cached_tokens
    
    def get_stats(self) -> dict:
        cache_rate = self.cached_tokens / self.total_input_tokens if self.total_input_tokens > 0 else 0
        return {
            "requests": self.requests,
            "total_input_tokens": self.total_input_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_hit_rate": f"{cache_rate:.2%}",
            "estimated_savings": f"${self.cached_tokens * 0.0000025:.4f}"  # 假设节省 $2.5/1M
        }

monitor = CacheMonitor()
```

## 最佳实践

1. **固定前缀**：把不变的内容放在消息开头
2. **合理分组**：相似请求放在一起发送，提高缓存命中
3. **监控命中率**：定期检查缓存效果
4. **避免动态内容**：时间戳、随机数等放在消息末尾
5. **批量处理**：相同上下文的请求批量发送

## 延伸阅读

- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)
- [Anthropic Prompt Caching](https://docs.anthropic.com/claude/docs/prompt-caching)
