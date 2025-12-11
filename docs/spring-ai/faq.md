---
sidebar_position: 11
title: 常见问题
---

# 常见问题 (FAQ)

## 基础问题

### Spring AI 目前是什么版本状态？

Spring AI 目前处于 **Milestone** 阶段，版本号为 1.0.0-M 系列。API 可能会在正式 GA 版本前变化。

### 如何添加 Spring AI 依赖？

需要添加 Spring 仓库：

```xml
<repositories>
    <repository>
        <id>spring-milestones</id>
        <url>https://repo.spring.io/milestone</url>
    </repository>
</repositories>

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
```

### Spring AI 支持哪些 Java 版本？

需要 **Java 17** 或更高版本。推荐使用 Java 21。

## 配置问题

### 如何切换不同的 AI 提供商？

1. 更换 starter 依赖
2. 更新配置文件

代码中的 `ChatClient` 无需修改，Spring AI 会自动注入对应实现。

### 如何处理 API Key 泄露风险？

1. 使用环境变量或密钥管理服务
2. 在生产环境使用权限受限的 API Key
3. 定期轮换 API Key

## 性能问题

### AI 请求响应很慢怎么办？

1. 使用流式响应
2. 减少 `maxTokens`
3. 使用更快的模型 (如 GPT-3.5)
4. 添加缓存
5. 考虑本地模型 (Ollama)

### 如何减少 API 调用成本？

1. 缓存响应
2. 限制 token 使用
3. 使用更便宜的模型
4. 监控使用量

## 常见错误

### `ChatClient` Bean 未找到

确保：

1. 添加了正确的 starter 依赖
2. 配置了必要的属性 (如 API Key)
3. Spring Boot 版本兼容 (推荐 3.2+)

### API Key 无效 (401)

1. 检查 API Key 是否正确
2. 确认账户有余额
3. 验证 API Key 权限

### Rate Limit 错误 (429)

实现指数退避重试：

```java
@Retryable(
    value = RateLimitException.class,
    maxAttempts = 5,
    backoff = @Backoff(delay = 1000, multiplier = 2)
)
public String chat(String message) {
    return chatClient.prompt().user(message).call().content();
}
```

## RAG 相关

### 向量数据库如何选择？

| 需求     | 推荐             |
| -------- | ---------------- |
| 快速原型 | Chroma           |
| 生产环境 | PGvector, Milvus |
| 云托管   | Pinecone         |

### 文档分块大小如何设置？

建议 500-1000 tokens，保留 50-100 tokens 重叠。
