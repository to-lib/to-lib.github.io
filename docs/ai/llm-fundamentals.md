---
sidebar_position: 2
title: 🧠 LLM 基础知识
description: 深入理解大语言模型 (LLM) 的核心概念、Transformer 架构、Token 计算、生成参数以及主流模型（如 GPT-4o, Claude 3.5, Llama 3）的对比。
keywords:
  [
    LLM 基础,
    Transformer,
    Token,
    上下文窗口,
    Temperature,
    Top-P,
    GPT-4o,
    Claude 3.5,
    Llama 3,
  ]
---

# LLM 基础知识

大型语言模型 (Large Language Model, LLM) 是一类基于深度学习的人工智能模型，能够理解和生成人类语言。本文介绍 LLM 的核心概念和工作原理。

## Transformer 架构

现代 LLM 几乎都基于 **Transformer** 架构，这是 2017 年 Google 在论文《Attention Is All You Need》中提出的。

### 核心组件

```
输入文本 → Tokenization → Embedding → Transformer Layers → 输出概率
```

| 组件                     | 作用                                       |
| ------------------------ | ------------------------------------------ |
| **Self-Attention**       | 捕捉序列中任意两个位置之间的依赖关系       |
| **Multi-Head Attention** | 并行运行多个注意力机制，捕捉不同类型的关系 |
| **Feed-Forward Network** | 对每个位置独立进行非线性变换               |
| **Layer Normalization**  | 稳定训练过程                               |

### 模型类型

| 类型                | 代表模型           | 特点                       |
| ------------------- | ------------------ | -------------------------- |
| **Encoder-only**    | BERT               | 双向理解，适合分类、NER    |
| **Decoder-only**    | GPT, LLaMA, Claude | 自回归生成，适合文本生成   |
| **Encoder-Decoder** | T5, BART           | 序列到序列，适合翻译、摘要 |

## Token 与上下文窗口

### 什么是 Token？

Token 是模型处理文本的基本单位。一个 token 可能是：

- 一个完整的单词：`hello`
- 单词的一部分：`un` + `believ` + `able`
- 一个中文字：`你` `好`
- 标点符号：`,` `.`

**经验法则**：

- 英文：1 token ≈ 4 个字符 或 0.75 个单词
- 中文：1 token ≈ 1-2 个汉字

### 上下文窗口 (Context Window)

上下文窗口是模型一次能处理的最大 token 数量。

| 模型           | 上下文长度 |
| -------------- | ---------- |
| GPT-4o         | 128K       |
| Claude 3.5     | 200K       |
| Gemini 1.5 Pro | 1M - 2M    |
| LLaMA 3.1      | 128K       |
| Qwen 2.5       | 32K - 128K |

:::tip 计算 Token
使用 tiktoken (OpenAI) 或 Hugging Face tokenizers 库计算 token 数量：

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, world!")
print(len(tokens))  # 4
```

:::

## 生成参数

### Temperature (温度)

控制输出的随机性。

| 值   | 效果                                 |
| ---- | ------------------------------------ |
| 0.0  | 确定性输出，总是选择最高概率的 token |
| 0.7  | 平衡创造性和一致性 (推荐默认值)      |
| 1.0+ | 高创造性，可能产生意外或不连贯的输出 |

### Top-P (核采样)

只从累积概率达到 P 的最小 token 集合中采样。

```
top_p=0.9 → 只考虑累积概率前 90% 的 tokens
```

### Max Tokens

限制生成的最大 token 数量。注意：输入 + 输出不能超过上下文窗口。

### 其他参数

| 参数                | 作用                           |
| ------------------- | ------------------------------ |
| `frequency_penalty` | 惩罚已出现过的 token，减少重复 |
| `presence_penalty`  | 惩罚已出现的主题，增加多样性   |
| `stop`              | 停止词序列，遇到时停止生成     |

## 主流模型对比

### 商业模型

| 模型                  | 提供商    | 特点                                    |
| --------------------- | --------- | --------------------------------------- |
| **GPT-4o**            | OpenAI    | 多模态，速度快，性价比极高，综合能力强  |
| **Claude 3.5 Sonnet** | Anthropic | 编码能力极强，逻辑推理出色，UI 设计友好 |
| **Gemini 1.5 Pro**    | Google    | 2M 超长上下文，原生多模态，生态集成好   |
| **DeepSeek V3**       | DeepSeek  | 国产之光，开源闭源皆强，编码与推理一流  |
| **Doubao (豆包)**     | 字节跳动  | 语音交互强，C 端应用广泛，API 价格低    |

### 开源模型

| 模型         | 参数量     | 特点                |
| ------------ | ---------- | ------------------- |
| **LLaMA 3**  | 8B / 70B   | Meta 出品，性能优秀 |
| **Mistral**  | 7B         | 小巧高效，适合部署  |
| **Qwen 2**   | 0.5B - 72B | 阿里出品，中英文佳  |
| **DeepSeek** | 7B / 67B   | 国产，代码能力强    |

## API 调用示例

### OpenAI API

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Anthropic API

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain transformers in simple terms."}
    ]
)

print(message.content[0].text)
```

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [OpenAI API 文档](https://platform.openai.com/docs)
- [Anthropic API 文档](https://docs.anthropic.com)
- [Hugging Face 模型库](https://huggingface.co/models)
