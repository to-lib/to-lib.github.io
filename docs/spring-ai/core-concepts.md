---
sidebar_position: 3
title: 核心概念
---

# 核心概念

深入了解 Spring AI 的关键组件和概念。

## Models (模型)

Spring AI 支持多种类型的 AI 模型：

- **Chat Models**: 用于对话式交互，输入文本，输出文本（通常是对话形式）。
- **Embedding Models**: 将文本转换为向量（数字列表），用于语义搜索和 RAG。
- **Image Models**: 根据文本描述生成图像 (例如 DALL-E)。

## Prompts (提示词)

Prompt 是与 AI 模型交互的基础。Spring AI 提供了 `Prompt` 类来封装发送给模型的消息。

### Prompt Template

类似于 Spring 的 `JdbcTemplate` 或 `RestTemplate`，`PromptTemplate` 允许你创建带有占位符的提示词：

```java
PromptTemplate promptTemplate = new PromptTemplate("Tell me a {adjective} joke about {topic}");
Prompt prompt = promptTemplate.create(Map.of("adjective", "funny", "topic", "cows"));
```

## Output Parsers (输出解析器)

AI 模型通常返回字符串。`OutputParser` 接口负责将这些字符串结构化为 Java 对象 (POJO)。

例如，使用 `BeanOutputParser`：

```java
record ActorsFilms(String actor, List<String> movies) {}

BeanOutputParser<ActorsFilms> parser = new BeanOutputParser<>(ActorsFilms.class);
String format = parser.getFormat();

PromptTemplate template = new PromptTemplate("""
    Generate the filmography for the actor {actor}.
    {format}
    """);
// ... 发送请求 ...
ActorsFilms actorsFilms = parser.parse(response);
```

## Vector Databases (向量数据库)

为了支持 RAG (Retrieval Augmented Generation)，Spring AI 提供了对向量数据库的抽象 `VectorStore`。它允许你存储和查询文档嵌入。

支持的实现包括：

- Chroma
- Milvus
- Pinecone
- Redis
- Neo4j
- PGvector
