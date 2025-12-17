---
sidebar_position: 8
title: 📋 快速参考
---

# AI 开发快速参考

本文汇总 AI 开发中常用的 API、参数、命令和代码片段，便于快速查阅。

## 模型参数速查

### 通用生成参数

| 参数                | 范围         | 推荐值   | 说明                   |
| ------------------- | ------------ | -------- | ---------------------- |
| `temperature`       | 0.0 - 2.0    | 0.7      | 控制随机性，越高越随机 |
| `top_p`             | 0.0 - 1.0    | 0.9      | 核采样阈值             |
| `max_tokens`        | 1 - 模型上限 | 按需设置 | 最大输出 token 数      |
| `frequency_penalty` | -2.0 - 2.0   | 0        | 惩罚重复 token         |
| `presence_penalty`  | -2.0 - 2.0   | 0        | 惩罚重复主题           |

### 场景推荐配置

| 场景     | temperature | top_p | 说明             |
| -------- | ----------- | ----- | ---------------- |
| 代码生成 | 0.0 - 0.2   | 0.9   | 确定性，减少错误 |
| 文档写作 | 0.5 - 0.7   | 0.9   | 平衡准确与流畅   |
| 创意写作 | 0.8 - 1.0   | 0.95  | 增加创意         |
| 数据提取 | 0.0         | 1.0   | 严格确定性       |
| 聊天对话 | 0.7         | 0.9   | 自然对话         |

## API 端点速查

### OpenAI

| 功能        | 端点                            |
| ----------- | ------------------------------- |
| Chat        | `POST /v1/chat/completions`     |
| Embeddings  | `POST /v1/embeddings`           |
| Images      | `POST /v1/images/generations`   |
| Audio (TTS) | `POST /v1/audio/speech`         |
| Audio (STT) | `POST /v1/audio/transcriptions` |

### Anthropic

| 功能     | 端点                |
| -------- | ------------------- |
| Messages | `POST /v1/messages` |

## 常用代码片段

### OpenAI Chat Completion

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### OpenAI Embedding

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Your text here"
)

embedding = response.data[0].embedding  # 1536 维向量
```

### Anthropic Messages

```python
import anthropic

client = anthropic.Anthropic(api_key="your-key")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(message.content[0].text)
```

### LangChain 基础

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!")
]

response = llm.invoke(messages)
print(response.content)
```

### LangChain RAG 简化版

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 切分文档
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 2. 存入向量库
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# 3. 检索
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("your question")

# 4. 生成回答
llm = ChatOpenAI(model="gpt-4o")
context = "\n".join([doc.page_content for doc in docs])
prompt = f"Context:\n{context}\n\nQuestion: your question\n\nAnswer:"
response = llm.invoke(prompt)
```

## Embedding 模型对比

| 模型                   | 提供商 | 维度 | 最大 Token | 特点         |
| ---------------------- | ------ | ---- | ---------- | ------------ |
| text-embedding-3-small | OpenAI | 1536 | 8191       | 经济实惠     |
| text-embedding-3-large | OpenAI | 3072 | 8191       | 高精度       |
| text-embedding-ada-002 | OpenAI | 1536 | 8191       | 旧版本       |
| bge-large-zh           | BAAI   | 1024 | 512        | 中文最佳开源 |
| m3e-base               | Moka   | 768  | 512        | 中文开源     |

## 向量数据库对比

| 数据库            | 类型 | 特点            |
| ----------------- | ---- | --------------- |
| **Pinecone**      | 托管 | 易用，自动扩展  |
| **Chroma**        | 开源 | 轻量，开发友好  |
| **Milvus**        | 开源 | 高性能，生产级  |
| **Qdrant**        | 开源 | Rust 实现，快速 |
| **pgvector**      | 扩展 | PostgreSQL 插件 |
| **Elasticsearch** | 扩展 | 混合检索        |

## Prompt 模板

### 通用助手

```
你是一个专业的 {领域} 助手。
请以简洁、准确的方式回答用户问题。
如果不确定答案，请诚实说明。

用户问题：{question}
```

### RAG 问答

```
请根据以下上下文回答问题。只使用上下文中的信息，不要编造。
如果上下文中没有相关信息，请说"我没有找到相关信息"。

上下文：
{context}

问题：{question}

回答：
```

### 结构化输出

```
请分析以下内容并以 JSON 格式输出结果。

内容：
{content}

输出格式：
{format_schema}
```

## 常用命令

### 安装依赖

```bash
# OpenAI
pip install openai

# Anthropic
pip install anthropic

# LangChain
pip install langchain langchain-openai langchain-community

# 向量数据库
pip install chromadb
pip install pinecone-client
```

### 环境变量

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Token 计算

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# 使用
tokens = count_tokens("Hello, world!")
print(f"Token 数量: {tokens}")
```

## 成本估算

| 模型              | 输入价格 | 输出价格  |
| ----------------- | -------- | --------- |
| GPT-4o            | $2.50/1M | $10.00/1M |
| GPT-4o-mini       | $0.15/1M | $0.60/1M  |
| Claude 3.5 Sonnet | $3.00/1M | $15.00/1M |
| Claude 3.5 Haiku  | $0.25/1M | $1.25/1M  |

_价格单位：美元/百万 token，数据更新于 2024 年_
