---
sidebar_position: 4
title: API 参考
---

# API 参考

Spring AI 的核心接口和类概览。

## ChatClient

最常用的接口，用于与聊天模型交互。

### 构建器模式

```java
ChatClient chatClient = ChatClient.builder(chatModel)
    .defaultSystem("你是一个专业助手")
    .defaultAdvisors(new LoggingAdvisor())
    .build();
```

### 核心方法

```java
// Fluent API 调用
chatClient.prompt()
    .system(systemPrompt)      // 系统提示词
    .user(userMessage)         // 用户消息
    .messages(messageList)     // 消息列表
    .functions(functionNames)  // 函数调用
    .advisors(advisors)        // Advisor
    .options(chatOptions)      // 模型参数
    .call()                    // 同步调用
    .stream();                 // 流式调用
```

### 响应获取

```java
// 获取内容
String content = chatClient.prompt().user(msg).call().content();

// 获取完整响应
ChatResponse response = chatClient.prompt().user(msg).call().chatResponse();

// 获取实体
Book book = chatClient.prompt().user(msg).call().entity(Book.class);
```

## ChatClient.Builder

用于创建和配置 ChatClient 实例。

| 方法                 | 说明              |
| -------------------- | ----------------- |
| `defaultSystem()`    | 默认系统提示词    |
| `defaultAdvisors()`  | 默认 Advisor 列表 |
| `defaultFunctions()` | 默认可用函数      |
| `defaultOptions()`   | 默认模型参数      |

```java
ChatClient client = ChatClient.builder(chatModel)
    .defaultSystem("你是助手")
    .defaultAdvisors(advisor1, advisor2)
    .defaultOptions(ChatOptionsBuilder.builder()
        .withTemperature(0.7f)
        .build())
    .build();
```

## StreamingChatClient

用于流式传输响应（例如打字机效果）。

```java
Flux<String> stream = chatClient.prompt()
    .user("讲个故事")
    .stream()
    .content();

// 完整流式响应
Flux<ChatResponse> responseStream = chatClient.prompt()
    .user("讲个故事")
    .stream()
    .chatResponse();
```

## ImageClient

用于图像生成。

```java
ImageResponse response = imageClient.call(
    new ImagePrompt(description, ImageOptionsBuilder.builder()
        .withModel("dall-e-3")
        .withQuality("hd")
        .withWidth(1024)
        .withHeight(1024)
        .build())
);

String url = response.getResult().getOutput().getUrl();
```

## EmbeddingClient

用于生成文本嵌入。

```java
// 单文本嵌入
List<Double> vector = embeddingClient.embed("文本内容");

// 批量嵌入
EmbeddingResponse response = embeddingClient.embedForResponse(List.of(
    "文本1", "文本2", "文本3"
));
```

## 数据类

### Prompt

包含一系列 `Message` 和 `ChatOptions`。

```java
Prompt prompt = new Prompt(
    List.of(
        new SystemMessage("你是助手"),
        new UserMessage("你好")
    ),
    ChatOptionsBuilder.builder().withTemperature(0.5f).build()
);
```

### Message 类型

| 类型               | 说明               |
| ------------------ | ------------------ |
| `UserMessage`      | 用户输入           |
| `SystemMessage`    | 系统指令（上下文） |
| `AssistantMessage` | AI 的回复          |
| `FunctionMessage`  | 函数调用结果       |

### ChatOptions

模型调用参数。

```java
ChatOptions options = ChatOptionsBuilder.builder()
    .withModel("gpt-4")
    .withTemperature(0.7f)
    .withMaxTokens(2000)
    .withTopP(0.9f)
    .withTopK(40)
    .withFrequencyPenalty(0.5f)
    .withPresencePenalty(0.5f)
    .build();
```

### ChatResponse

模型响应封装。

```java
ChatResponse response = chatClient.prompt().user(msg).call().chatResponse();

// 获取生成结果
Generation generation = response.getResult();
String content = generation.getOutput().getContent();

// 获取元数据
ChatResponseMetadata metadata = response.getMetadata();
Usage usage = metadata.getUsage();
int promptTokens = usage.getPromptTokens();
int completionTokens = usage.getGenerationTokens();
```

## 向量存储

### VectorStore

向量数据库抽象接口。

```java
// 添加文档
vectorStore.add(List.of(
    new Document("内容1", Map.of("source", "file1.txt")),
    new Document("内容2")
));

// 相似度搜索
List<Document> results = vectorStore.similaritySearch(
    SearchRequest.query("查询文本")
        .withTopK(5)
        .withSimilarityThreshold(0.7)
        .withFilterExpression("source == 'file1.txt'")
);

// 删除文档
vectorStore.delete(List.of("doc-id-1", "doc-id-2"));
```

### Document

文档封装类。

```java
Document doc = new Document(
    "文档内容",
    Map.of(
        "source", "manual.pdf",
        "page", 5,
        "author", "张三"
    )
);

String id = doc.getId();
String content = doc.getContent();
Map<String, Object> metadata = doc.getMetadata();
List<Double> embedding = doc.getEmbedding();
```

### SearchRequest

搜索请求构建器。

```java
SearchRequest request = SearchRequest.query("查询内容")
    .withTopK(10)                                    // 返回数量
    .withSimilarityThreshold(0.75)                   // 相似度阈值
    .withFilterExpression("year >= 2024 && type == 'article'");  // 过滤条件
```

## Advisor 接口

### RequestResponseAdvisor

```java
public interface RequestResponseAdvisor {
    default AdvisedRequest adviseRequest(AdvisedRequest request, Map<String, Object> context) {
        return request;
    }

    default ChatResponse adviseResponse(ChatResponse response, Map<String, Object> context) {
        return response;
    }
}
```

## 异常类

| 异常类                    | 说明           | 可重试 |
| ------------------------- | -------------- | :----: |
| `AiException`             | AI 异常基类    |   -    |
| `NonTransientAiException` | 不可重试错误   |   ❌   |
| `TransientAiException`    | 可重试临时错误 |   ✅   |
| `RateLimitException`      | API 限流       |   ✅   |
| `ContentPolicyException`  | 内容违规       |   ❌   |

---

更多详细信息请参考 [Spring AI 官方文档](https://spring.io/projects/spring-ai)。
