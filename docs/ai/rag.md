---
sidebar_position: 4
title: 📚 RAG (检索增强生成)
---

# RAG (检索增强生成)

**RAG (Retrieval-Augmented Generation)** 是一种结合了**检索 (Retrieval)** 和 **生成 (Generation)** 的 AI 技术架构。它通过在生成回答之前先从外部知识库中检索相关信息，并将其作为上下文输入给大型语言模型 (LLM)，从而显著提升回答的准确性和时效性。

## 为什么需要 RAG？

LLM (如 GPT-4) 存在以下局限性：

- **知识截止**：模型训练数据是静态的，无法获知最新的时事。
- **幻觉 (Hallucination)**：在不知道答案时可能会一本正经地胡说八道。
- **私有数据缺失**：模型从未见过企业的内部文档和私有数据。

RAG 通过外挂知识库通过解决了这些问题。

## RAG 的工作流程

RAG 的典型流程包含三个阶段：

### 1. 索引 (Indexing) - 准备阶段

将文档转换为向量并存入数据库。

- **加载 (Load)**：读取 PDF, Word, Markdown, HTML 等文件。
- **切分 (Split)**：将长文档切分为较小的文本块 (Chunks)。
- **嵌入 (Embed)**：使用 Embedding 模型将文本块转换为向量 (Vectors)。
- **存储 (Store)**：将向量存储到向量数据库 (Vector DB) 中。

### 2. 检索 (Retrieval) - 运行阶段

- **查询编码**：将用户的自然语言问题转换为向量。
- **相似度搜索**：在向量数据库中查找与问题向量最相似的文本块 (Top-K)。

### 3. 生成 (Generation) - 运行阶段

- **构建 Prompt**：将检索到的文本块作为“上下文 (Context)”填入 Prompt 模板。
- **LLM 回答**：LLM 基于提供的上下文回答用户的问题。

## 核心技术栈

### 向量数据库 (Vector Database)

- **Pinecone**: 托管型向量数据库，易于使用。
- **Milvus**: 开源高性能向量数据库。
- **Chroma**: 轻量级开源向量数据库，适合本地开发。
- **Elasticsearch / pgvector**: 传统数据库的向量扩展。

### 开发框架

- **LangChain**: 最流行的 LLM 应用开发框架，提供了丰富的 RAG 组件。
- **LlamaIndex**: 专注于数据索引和检索的框架，对 RAG 优化极佳。

## 高级 RAG 技巧

- **混合检索 (Hybrid Search)**：结合关键词检索 (BM25) 和向量检索，提高召回率。
- **重排序 (Re-ranking)**：检索出较多结果后，使用专门的 Re-rank 模型进行精细排序。
- **元数据过滤 (Metadata Filtering)**：在检索前通过时间、作者等标签过滤数据。
- **查询重写 (Query Rewriting)**：将用户模糊的问题改写为更适合检索的形式。
