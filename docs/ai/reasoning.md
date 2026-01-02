---
sidebar_position: 25
title: 🧩 Reasoning 模型
---

# Reasoning 模型（推理模型）

Reasoning 模型（如 OpenAI o1/o3 系列）是专门为复杂推理任务设计的模型，通过"思考"过程来解决数学、编程、科学等需要深度推理的问题。

## 什么是 Reasoning 模型？

传统 LLM 直接生成答案，而 Reasoning 模型会先进行内部推理（Chain of Thought），然后给出答案。

```
传统模型：
问题 ──────────────────────────────> 答案

Reasoning 模型：
问题 ───> [内部推理过程] ───> 答案
          (不可见/部分可见)
```

## 模型对比

| 模型    | 特点                     | 适用场景           | 价格（输入/输出）  |
| ------- | ------------------------ | ------------------ | ------------------ |
| o1      | 最强推理能力             | 复杂数学、科学研究 | $15/$60 per 1M     |
| o1-mini | 平衡推理能力和速度       | 编程、一般推理     | $3/$12 per 1M      |
| o1-pro  | 更长思考时间，更高准确率 | 最复杂问题         | 按需定价           |
| o3      | 下一代推理模型           | 前沿研究           | 待发布             |

## 基础使用

### OpenAI o1 API

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="o1",  # 或 "o1-mini"
    messages=[
        {
            "role": "user",
            "content": "证明：对于任意正整数 n，n³ - n 能被 6 整除。"
        }
    ]
    # 注意：o1 不支持 temperature、top_p 等参数
    # 也不支持 system message
)

print(response.choices[0].message.content)
```

### 查看推理 Token

```python
response = client.chat.completions.create(
    model="o1",
    messages=[{"role": "user", "content": "解决这个问题..."}]
)

# 查看 token 使用
usage = response.usage
print(f"输入 tokens: {usage.prompt_tokens}")
print(f"输出 tokens: {usage.completion_tokens}")
print(f"推理 tokens: {usage.completion_tokens_details.reasoning_tokens}")
```

## o1 与 GPT-4o 的区别

| 特性           | o1                 | GPT-4o             |
| -------------- | ------------------ | ------------------ |
| System Message | ❌ 不支持          | ✅ 支持            |
| Temperature    | ❌ 固定为 1        | ✅ 可调节          |
| Streaming      | ❌ 不支持          | ✅ 支持            |
| Function Call  | ✅ 支持            | ✅ 支持            |
| 图像输入       | ✅ 支持（o1）      | ✅ 支持            |
| 响应速度       | 较慢（需要思考）   | 较快               |
| 推理能力       | 强                 | 一般               |
| 成本           | 高                 | 中                 |

## 适用场景

### ✅ 适合使用 o1 的场景

1. **复杂数学问题**
```python
response = client.chat.completions.create(
    model="o1",
    messages=[{
        "role": "user",
        "content": """
        求解微分方程：y'' + 4y' + 4y = e^(-2x)
        给出通解和特解。
        """
    }]
)
```

2. **算法设计**
```python
response = client.chat.completions.create(
    model="o1-mini",
    messages=[{
        "role": "user",
        "content": """
        设计一个算法，在 O(n log n) 时间复杂度内找出数组中
        所有和为目标值的三元组，不能有重复。
        """
    }]
)
```

3. **代码调试与优化**
```python
response = client.chat.completions.create(
    model="o1-mini",
    messages=[{
        "role": "user",
        "content": f"""
        这段代码有一个微妙的 bug，请找出并修复：
        
        ```python
        {buggy_code}
        ```
        
        错误现象：{error_description}
        """
    }]
)
```

4. **科学推理**
```python
response = client.chat.completions.create(
    model="o1",
    messages=[{
        "role": "user",
        "content": """
        分析以下实验数据，推断可能的化学反应机理：
        [实验数据...]
        """
    }]
)
```

### ❌ 不适合使用 o1 的场景

- 简单问答（用 GPT-4o-mini）
- 创意写作（用 GPT-4o）
- 需要流式输出的场景
- 成本敏感的场景
- 需要快速响应的场景

## 最佳实践

### 1. 提供清晰的问题描述

```python
# ✅ 好的提示
prompt = """
问题：一个袋子里有 5 个红球和 3 个蓝球。
不放回地抽取 3 个球，求至少有 2 个红球的概率。

请：
1. 列出所有可能的情况
2. 计算每种情况的概率
3. 给出最终答案
"""

# ❌ 不好的提示
prompt = "算一下概率"
```

### 2. 分步骤要求

```python
prompt = """
请解决以下编程问题：

问题：实现一个 LRU 缓存

要求：
1. 首先分析问题，确定数据结构
2. 设计算法，说明时间复杂度
3. 编写代码
4. 给出测试用例
5. 分析边界情况
"""
```

### 3. 提供约束条件

```python
prompt = """
设计一个分布式锁的实现方案。

约束条件：
- 必须保证互斥性
- 需要处理死锁情况
- 支持可重入
- 考虑网络分区场景
- 使用 Redis 作为存储

请给出详细的设计方案和伪代码。
"""
```

## 与其他模型配合

### 路由策略

```python
def route_to_model(query: str) -> str:
    """根据问题复杂度选择模型"""
    
    # 使用小模型判断复杂度
    complexity_check = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""判断以下问题的复杂度（1-5分）：
            
问题：{query}

只返回数字。"""
        }],
        max_tokens=10
    )
    
    try:
        complexity = int(complexity_check.choices[0].message.content.strip())
    except:
        complexity = 3
    
    if complexity >= 4:
        return "o1"
    elif complexity >= 3:
        return "o1-mini"
    else:
        return "gpt-4o-mini"

# 使用
model = route_to_model(user_query)
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": user_query}]
)
```

### 两阶段处理

```python
def two_stage_reasoning(problem: str) -> str:
    """两阶段处理：o1 推理 + GPT-4o 润色"""
    
    # 阶段 1：使用 o1 进行推理
    reasoning_response = client.chat.completions.create(
        model="o1-mini",
        messages=[{"role": "user", "content": problem}]
    )
    
    raw_answer = reasoning_response.choices[0].message.content
    
    # 阶段 2：使用 GPT-4o 润色输出
    polished_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "请将以下技术内容整理成清晰、易读的格式，保持准确性。"
            },
            {"role": "user", "content": raw_answer}
        ]
    )
    
    return polished_response.choices[0].message.content
```

## 成本优化

### 1. 问题预筛选

```python
def needs_reasoning(query: str) -> bool:
    """判断是否需要推理模型"""
    keywords = ["证明", "推导", "算法", "优化", "分析", "设计", "debug"]
    return any(kw in query for kw in keywords)

def smart_query(query: str) -> str:
    if needs_reasoning(query):
        model = "o1-mini"
    else:
        model = "gpt-4o-mini"
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content
```

### 2. 缓存推理结果

```python
import hashlib
import json

class ReasoningCache:
    def __init__(self):
        self.cache = {}
    
    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> str | None:
        key = self._hash_query(query)
        return self.cache.get(key)
    
    def set(self, query: str, response: str):
        key = self._hash_query(query)
        self.cache[key] = response
    
    def query(self, query: str) -> str:
        # 检查缓存
        cached = self.get(query)
        if cached:
            return cached
        
        # 调用 API
        response = client.chat.completions.create(
            model="o1-mini",
            messages=[{"role": "user", "content": query}]
        )
        
        result = response.choices[0].message.content
        self.set(query, result)
        return result
```

## 实战示例

### 代码审查

```python
def code_review_with_reasoning(code: str, language: str = "Python") -> str:
    """使用推理模型进行深度代码审查"""
    
    prompt = f"""
请对以下 {language} 代码进行深度审查：

```{language.lower()}
{code}
```

请分析：
1. 潜在的 bug 和边界情况
2. 性能问题和优化建议
3. 安全漏洞
4. 代码风格和最佳实践
5. 可能的重构方向

对于每个问题，请说明原因和修复建议。
"""
    
    response = client.chat.completions.create(
        model="o1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

### 数学证明

```python
def mathematical_proof(statement: str) -> str:
    """数学证明"""
    
    prompt = f"""
请证明以下数学命题：

{statement}

要求：
1. 给出严格的数学证明
2. 说明使用的定理和引理
3. 如果有多种证明方法，给出最优雅的一种
4. 解释证明的关键步骤
"""
    
    response = client.chat.completions.create(
        model="o1",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

## 延伸阅读

- [OpenAI o1 文档](https://platform.openai.com/docs/guides/reasoning)
- [o1 System Card](https://openai.com/index/openai-o1-system-card/)
