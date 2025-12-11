---
sidebar_position: 12
title: 面试题精选
---

# Spring AI 面试题精选

## 基础概念

### 1. 什么是 Spring AI？它解决了什么问题？

**答案**：Spring AI 是 Spring 生态中用于构建 AI 驱动应用的框架。它解决了：

- AI 模型接入的复杂性
- 不同 AI 提供商 API 的差异
- 企业级 AI 应用的工程化需求

核心价值是提供统一的抽象层，使切换 AI 提供商像切换数据库一样简单。

### 2. Spring AI 的核心组件有哪些？

**答案**：

- `ChatClient`: 聊天模型客户端
- `EmbeddingClient`: 文本嵌入客户端
- `ImageClient`: 图像生成客户端
- `VectorStore`: 向量数据库抽象
- `PromptTemplate`: 提示词模板
- `OutputParser`: 输出解析器

### 3. 什么是 RAG？Spring AI 如何支持 RAG？

**答案**：RAG (Retrieval Augmented Generation) 是通过检索相关文档来增强 AI 回答的技术。

Spring AI 通过以下组件支持 RAG：

- `VectorStore` 接口及多种实现 (PGvector, Milvus 等)
- `EmbeddingClient` 生成文本嵌入
- 文档读取器和分块器

## 实践问题

### 4. 如何实现多轮对话？

**答案**：维护消息历史列表：

```java
List<Message> history = new ArrayList<>();

public String chat(String userInput) {
    history.add(new UserMessage(userInput));

    ChatResponse response = chatClient.prompt()
            .messages(history)
            .call()
            .chatResponse();

    String reply = response.getResult().getOutput().getContent();
    history.add(new AssistantMessage(reply));
    return reply;
}
```

### 5. 如何将 AI 输出解析为 Java 对象？

**答案**：使用 `BeanOutputParser`：

```java
record Book(String title, String author) {}

BeanOutputParser<Book> parser = new BeanOutputParser<>(Book.class);
String prompt = "推荐一本书。" + parser.getFormat();

String response = chatClient.prompt().user(prompt).call().content();
Book book = parser.parse(response);
```

### 6. Spring AI 如何实现 Function Calling？

**答案**：

1. 定义 `@Bean` 方法返回 `Function`
2. 使用 `@Description` 注解描述函数用途
3. 在调用时通过 `.functions()` 指定可用函数

```java
@Bean
@Description("获取天气")
public Function<WeatherRequest, WeatherResponse> getWeather() {
    return req -> weatherService.get(req.city());
}
```

### 7. 如何处理 AI API 的限流和错误？

**答案**：

- 使用 Spring Retry 实现重试
- 使用 Resilience4j 实现熔断和限流
- 区分 `TransientAiException` (可重试) 和 `NonTransientAiException` (不可重试)

### 8. 本地部署 AI 模型有什么优势？如何实现？

**答案**：优势：

- 数据隐私，无外泄风险
- 无 API 调用成本
- 低延迟

实现：使用 Ollama + Spring AI Ollama Starter

```yaml
spring:
  ai:
    ollama:
      base-url: http://localhost:11434
      chat:
        options:
          model: llama2
```

## 架构设计

### 9. 如何设计一个生产级的 AI 应用？

**答案**：关键考虑：

1. **安全**: API Key 管理、输入验证、Prompt 注入防护
2. **可靠性**: 重试、熔断、降级
3. **成本控制**: Token 监控、缓存、模型选择
4. **可观测性**: 日志、指标、追踪
5. **测试**: Mock 客户端、集成测试

### 10. Spring AI 与 LangChain4j 如何选择？

**答案**：
| 方面 | Spring AI | LangChain4j |
|------|-----------|-------------|
| 集成 | Spring 原生 | 独立框架 |
| 配置 | 自动配置 | 手动配置 |
| Agent | 基础 | 丰富 |
| 成熟度 | 较新 | 相对成熟 |

Spring 项目优先选择 Spring AI；需要复杂 Agent 功能考虑 LangChain4j。
