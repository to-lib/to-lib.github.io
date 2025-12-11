---
sidebar_position: 4
title: API 参考
---

# API 参考

Spring AI 的核心接口和类概览。

## 客户端接口

### `ChatClient`

最常用的接口，用于与聊天模型交互。

- `String call(String message)`: 发送简单消息并获取响应。
- `ChatResponse call(Prompt prompt)`: 发送结构化 Prompt 并获取完整响应（包含元数据）。

### `StreamingChatClient`

用于流式传输响应（例如打字机效果）。

- `Flux<ChatResponse> stream(Prompt prompt)`

### `ImageClient`

用于图像生成。

- `ImageResponse call(ImagePrompt prompt)`

### `EmbeddingClient`

用于生成文本嵌入。

- `List<Double> embed(String text)`
- `EmbeddingResponse embedForResponse(List<String> texts)`

## 数据类

### `Prompt`

包含一系列 `Message` 和 `ChatOptions`。

### `Message`

- `UserMessage`: 用户输入。
- `SystemMessage`: 系统指令（上下文设置）。
- `AssistantMessage`: AI 的回复。

## 向量存储

### `VectorStore`

- `void add(List<Document> documents)`: 添加文档。
- `List<Document> similaritySearch(String query)`: 相似度搜索。

---

更多详细信息请参考 [Spring AI 官方文档](https://spring.io/projects/spring-ai)。
