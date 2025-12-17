---
sidebar_position: 17
title: 快速参考
---

# 快速参考

Spring AI 常用配置和 API 速查表。

## 依赖配置

### Maven BOM

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.ai</groupId>
            <artifactId>spring-ai-bom</artifactId>
            <version>1.0.0-M4</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

<repositories>
    <repository>
        <id>spring-milestones</id>
        <url>https://repo.spring.io/milestone</url>
    </repository>
</repositories>
```

### 常用 Starter

| Starter                                        | 用途                 |
| ---------------------------------------------- | -------------------- |
| `spring-ai-openai-spring-boot-starter`         | OpenAI (GPT, DALL-E) |
| `spring-ai-azure-openai-spring-boot-starter`   | Azure OpenAI         |
| `spring-ai-ollama-spring-boot-starter`         | Ollama 本地模型      |
| `spring-ai-anthropic-spring-boot-starter`      | Anthropic Claude     |
| `spring-ai-bedrock-ai-spring-boot-starter`     | Amazon Bedrock       |
| `spring-ai-pgvector-store-spring-boot-starter` | PGvector 向量数据库  |
| `spring-ai-chroma-store-spring-boot-starter`   | Chroma 向量数据库    |
| `spring-ai-milvus-store-spring-boot-starter`   | Milvus 向量数据库    |

## 配置速查

### OpenAI

```yaml
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
      chat:
        options:
          model: gpt-4 # gpt-4, gpt-3.5-turbo
          temperature: 0.7
          max-tokens: 2000
      embedding:
        options:
          model: text-embedding-3-small
      image:
        options:
          model: dall-e-3
```

### Ollama

```yaml
spring:
  ai:
    ollama:
      base-url: http://localhost:11434
      chat:
        options:
          model: llama2 # llama2, mistral, codellama
          temperature: 0.7
      embedding:
        options:
          model: nomic-embed-text
```

### Azure OpenAI

```yaml
spring:
  ai:
    azure:
      openai:
        api-key: ${AZURE_OPENAI_API_KEY}
        endpoint: https://your-resource.openai.azure.com/
        deployment-name: gpt-4-deployment
```

## ChatClient API

### 基础调用

```java
// 简单调用
String response = chatClient.prompt()
    .user("你好")
    .call()
    .content();

// 流式调用
Flux<String> stream = chatClient.prompt()
    .user("讲个故事")
    .stream()
    .content();
```

### 带系统提示

```java
chatClient.prompt()
    .system("你是一个专业的技术顾问")
    .user(userMessage)
    .call()
    .content();
```

### 多轮对话

```java
chatClient.prompt()
    .messages(List.of(
        new SystemMessage("你是助手"),
        new UserMessage("问题1"),
        new AssistantMessage("回答1"),
        new UserMessage("问题2")
    ))
    .call()
    .content();
```

### 函数调用

```java
chatClient.prompt()
    .user("北京天气如何？")
    .functions("getWeather")
    .call()
    .content();
```

### 带选项

```java
chatClient.prompt()
    .user(message)
    .options(ChatOptionsBuilder.builder()
        .withTemperature(0.7f)
        .withMaxTokens(500)
        .withTopP(0.9f)
        .build())
    .call()
    .content();
```

### 使用 Advisor

```java
chatClient.prompt()
    .user(message)
    .advisors(new QuestionAnswerAdvisor(vectorStore))
    .call()
    .content();
```

## 输出解析

### Bean 解析

```java
record Book(String title, String author) {}

BeanOutputParser<Book> parser = new BeanOutputParser<>(Book.class);
String prompt = "推荐一本书。" + parser.getFormat();

String response = chatClient.prompt().user(prompt).call().content();
Book book = parser.parse(response);
```

### List 解析

```java
ListOutputParser parser = new ListOutputParser(new DefaultConversionService());
String prompt = "列出5个点子。" + parser.getFormat();

String response = chatClient.prompt().user(prompt).call().content();
List<String> ideas = parser.parse(response);
```

## VectorStore API

### 添加文档

```java
vectorStore.add(List.of(
    new Document("文档内容1"),
    new Document("文档内容2", Map.of("source", "file1.txt"))
));
```

### 相似度搜索

```java
List<Document> results = vectorStore.similaritySearch(
    SearchRequest.query("查询内容")
        .withTopK(5)
        .withSimilarityThreshold(0.7)
);
```

### 带过滤条件

```java
vectorStore.similaritySearch(
    SearchRequest.query("查询")
        .withTopK(5)
        .withFilterExpression("source == 'docs' && year >= 2024")
);
```

## 常用参数

### ChatOptions

| 参数          | 类型    | 说明           | 默认值     |
| ------------- | ------- | -------------- | ---------- |
| `model`       | String  | 模型名称       | 提供商默认 |
| `temperature` | Float   | 随机性 (0-1)   | 0.7        |
| `maxTokens`   | Integer | 最大输出 token | 无限制     |
| `topP`        | Float   | 核采样 (0-1)   | 1.0        |
| `topK`        | Integer | 候选词数量     | -          |

### ImageOptions

| 参数             | 类型   | 说明     | 可选值               |
| ---------------- | ------ | -------- | -------------------- |
| `model`          | String | 模型     | dall-e-2, dall-e-3   |
| `size`           | String | 尺寸     | 1024x1024, 1792x1024 |
| `quality`        | String | 质量     | standard, hd         |
| `style`          | String | 风格     | vivid, natural       |
| `responseFormat` | String | 返回格式 | url, b64_json        |

## 异常类型

| 异常                      | 说明         | 处理方式      |
| ------------------------- | ------------ | ------------- |
| `NonTransientAiException` | 不可重试错误 | 检查配置/输入 |
| `TransientAiException`    | 可重试错误   | 重试          |
| `RateLimitException`      | 限流错误     | 延迟重试      |
| `ContentPolicyException`  | 内容违规     | 修改输入      |

## Ollama 常用命令

```bash
# 安装
brew install ollama

# 启动服务
ollama serve

# 拉取模型
ollama pull llama2
ollama pull mistral
ollama pull codellama
ollama pull nomic-embed-text

# 查看已安装模型
ollama list

# 删除模型
ollama rm llama2
```

## 下一步

- [核心概念](/docs/spring-ai/core-concepts) - 深入了解核心概念
- [最佳实践](/docs/spring-ai/best-practices) - 生产环境最佳实践
