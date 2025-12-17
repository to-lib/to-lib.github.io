---
sidebar_position: 1
title: 🤖 AI 开发概览
---

# AI 开发概览

欢迎来到 AI 开发文档。这里汇集了关于人工智能应用开发的核心概念和技术指南，特别是针对大语言模型（LLM）的应用开发。

## 文档导航

### 📖 基础知识

- [🧠 LLM 基础知识](./llm-fundamentals) - Transformer 架构、Token、生成参数、主流模型介绍
- [✨ 提示工程](./prompt-engineering) - Prompt 设计原则、技巧与模板

### 🎯 核心技术

- [🤖 AI Agent (智能体)](./agent) - Agent 架构、工作模式与代码实现
- [🔧 Function Calling](./function-calling) - 函数调用原理与 API 使用
- [📚 RAG (检索增强生成)](./rag) - RAG 工作流程与代码实践
- [🔌 MCP (模型上下文协议)](./mcp) - 模型与外部工具的标准化连接协议

### 📚 参考指南

- [📋 快速参考](./quick-reference) - API、参数、代码片段速查
- [❓ 常见问题](./faq) - FAQ 解答

## 学习路线

```mermaid
graph LR
    A[LLM 基础] --> B[提示工程]
    B --> C[Function Calling]
    C --> D[RAG]
    D --> E[Agent]
    E --> F[MCP]
```

**推荐顺序**：

1. **基础概念**：先掌握 LLM 工作原理和 Prompt 技巧
2. **工具调用**：学习 Function Calling，让模型具备行动能力
3. **知识增强**：通过 RAG 连接外部知识库
4. **智能体**：构建完整的 Agent 系统
5. **标准集成**：使用 MCP 实现标准化工具连接

## 技术栈推荐

| 类型       | 推荐                          | 备选               |
| ---------- | ----------------------------- | ------------------ |
| **框架**   | LangChain                     | LlamaIndex         |
| **模型**   | GPT-4o / Claude 3.5           | Qwen2 / LLaMA 3    |
| **向量库** | Chroma (开发) / Milvus (生产) | Pinecone, pgvector |
| **评估**   | Langsmith                     | Arize, Deepeval    |
