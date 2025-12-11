---
sidebar_position: 1
title: Spring AI 概览
---

# Spring AI 概览

Spring AI 是一个用于构建 AI 驱动的应用的框架，它提供了一套统一的 API 来与各种 AI 模型进行交互，包括聊天模型、图像生成模型、嵌入模型等。

## 什么是 Spring AI？

Spring AI 旨在简化 AI 功能在 Java 企业级应用中的集成。它借鉴了 Spring 生态系统的设计原则，如可移植性和模块化设计，使开发者能够轻松地切换不同的 AI 提供商（如 OpenAI、Azure OpenAI、Ollama 等），而无需重写大量代码。

## 核心特性

- **统一的 API**: 提供通用的 `ChatClient`, `ImageClient`, `EmbeddingClient` 等接口。
- **多模型支持**: 支持 OpenAI, Azure OpenAI, Amazon Bedrock, Ollama, Hugging Face 等。
- **Prompt Engineering**: 支持 Prompt 模板和通过类似于 Spring MVC 的方式传递参数。
- **Output Parsing**: 自动将 AI 模型的输出映射为 Java 对象 (POJO)。
- **RAG (检索增强生成)**: 内置对向量数据库（Vector Databases）的支持，简化 RAG 应用的开发。

## 适用场景

- **智能客服**: 构建能够理解上下文并回答问题的聊天机器人。
- **内容生成**: 自动生成文章、摘要、代码或图像。
- **语义搜索**: 利用向量嵌入实现更精准的语义搜索。
- **文档分析**: 从大量文档中提取关键信息。

## 下一步

- [快速开始](./quick-start)
- [核心概念](./core-concepts)
