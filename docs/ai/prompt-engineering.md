---
sidebar_position: 3
title: ✨ 提示工程
description: 全面解析 Prompt Engineering 核心原则与技巧（Zero-shot, Few-shot, CoT），提供实用的 Prompt 模板与调试指南，助你掌握 LLM 交互艺术。
keywords:
  [
    提示工程,
    Prompt Engineering,
    Prompt 技巧,
    CoT,
    Few-shot,
    提示词模板,
    LLM 交互,
  ]
---

# 提示工程 (Prompt Engineering)

提示工程是设计和优化输入给大语言模型的提示词 (Prompt)，以获得期望输出的技术。良好的 Prompt 设计能显著提升 LLM 的回答质量。

## 基本原则

### 1. 清晰明确

```markdown
❌ 模糊: "帮我写点东西"
✅ 明确: "请用 Python 写一个函数，接收一个整数列表，返回其中所有偶数的平方和"
```

### 2. 提供上下文

```markdown
❌ 无上下文: "这段代码有什么问题？"
✅ 有上下文: "我在用 Python 3.11 处理 CSV 文件时遇到编码错误，这段代码有什么问题？[代码]"
```

### 3. 指定输出格式

```markdown
请分析以下文本的情感，输出格式为 JSON：
{
"sentiment": "positive/negative/neutral",
"confidence": 0.0-1.0,
"keywords": ["关键词列表"]
}
```

### 4. 使用分隔符

使用 `"""`, `###`, `---`, `<>` 等分隔不同部分的内容。

```markdown
请翻译以下被三引号包围的文本为中文：

"""
Artificial intelligence is transforming industries worldwide.
"""
```

## 常用技巧

### Zero-shot Prompting

直接描述任务，不提供示例。

```markdown
将以下句子翻译成法语：
"The weather is beautiful today."
```

### Few-shot Prompting

提供几个示例，引导模型理解任务模式。

```markdown
请按照示例分类以下评论的情感：

评论: "这个产品太棒了！" → 情感: 正面
评论: "质量很差，退货了" → 情感: 负面
评论: "还行吧，一般般" → 情感: 中性

评论: "超出预期，强烈推荐！" → 情感:
```

### Chain of Thought (CoT)

引导模型逐步思考，适用于复杂推理任务。

```markdown
问题：一个商店有 23 个苹果，卖出了 7 个，又进货了 12 个。现在有多少个苹果？

让我们一步步思考：

1. 初始苹果数量：23 个
2. 卖出后：23 - 7 = 16 个
3. 进货后：16 + 12 = 28 个

答案：28 个苹果
```

:::tip Zero-shot CoT
只需在问题后加上 "Let's think step by step" 即可激活思维链：

```
问题：...
让我们一步步思考。
```

:::

### Role Playing (角色扮演)

为模型设定特定角色和行为准则。

```markdown
你是一位资深的 Python 开发专家，拥有 10 年的后端开发经验。
请以导师的身份回答问题，耐心解释概念，并提供最佳实践建议。
当看到潜在问题时，主动指出并给出改进建议。
```

### 自我一致性 (Self-Consistency)

多次采样，选择多数一致的答案。适用于需要高可靠性的场景。

```python
responses = []
for _ in range(5):
    response = llm.generate(prompt, temperature=0.7)
    responses.append(response)

# 选择出现最多的答案
final_answer = most_common(responses)
```

## Prompt 模板

### 通用结构

```markdown
# 角色定义

你是 [角色描述]。

# 任务背景

[提供必要的上下文信息]

# 具体任务

请完成以下任务：
[详细的任务描述]

# 输出要求

- 格式：[指定格式]
- 长度：[长度限制]
- 语言：[语言要求]

# 输入内容

"""
[用户输入]
"""
```

### 代码生成模板

```markdown
请用 [编程语言] 编写一个 [功能描述] 的 [函数/类/脚本]。

要求：

- 输入：[输入参数描述]
- 输出：[输出描述]
- 约束：[边界条件或特殊要求]

请包含：

- 完整的类型注解
- 详细的文档字符串
- 边界情况处理
- 示例用法
```

### 文档总结模板

```markdown
请阅读以下文档并生成摘要。

文档内容：
"""
[文档内容]
"""

摘要要求：

- 长度：不超过 200 字
- 包含：主要观点、关键数据、结论
- 格式：分点列出
```

## 常见反模式

| 反模式   | 问题       | 改进                     |
| -------- | ---------- | ------------------------ |
| 过于模糊 | 输出不可控 | 提供具体细节和约束       |
| 信息过载 | 模型混淆   | 分解为多个简单任务       |
| 负面指令 | 效果差     | 使用正面指令描述期望行为 |
| 假设已知 | 输出偏差   | 明确提供所有必要上下文   |

```markdown
❌ "不要写得太长"
✅ "请限制在 100 字以内"

❌ "不要用技术术语"
✅ "请用适合初中生理解的语言解释"
```

## 高级技巧

### ReAct 模式

结合推理 (Reasoning) 和行动 (Acting)：

```markdown
问题：2024 年诺贝尔物理学奖获得者是谁？

思考：这是一个需要查询最新信息的问题，我需要搜索相关资料。
行动：[搜索] "2024 Nobel Prize Physics winner"
观察：[搜索结果]
思考：根据搜索结果，我可以回答这个问题。
答案：...
```

### 结构化输出

要求模型输出特定格式的数据：

````markdown
分析以下产品评论，以 JSON 格式输出结果。

评论："""[评论内容]"""

输出格式：

```json
{
  "summary": "一句话总结",
  "pros": ["优点列表"],
  "cons": ["缺点列表"],
  "rating": 1-5,
  "recommendation": true/false
}
```
````

## 调试技巧

1. **从简单开始**：先用最简单的 prompt 测试，逐步添加复杂性
2. **迭代优化**：根据输出结果不断调整 prompt
3. **A/B 测试**：对比不同版本的 prompt 效果
4. **记录版本**：保存每个版本的 prompt 和对应效果
5. **理解失败**：分析失败案例，找出问题根源

## 延伸阅读

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Learn Prompting](https://learnprompting.org/)
