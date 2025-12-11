---
sidebar_position: 10
title: 最佳实践
---

# 最佳实践

在生产环境中使用 Spring AI 的推荐做法。

## API Key 管理

### 使用环境变量

```yaml
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
```

### 使用 Spring Cloud Config 或 Vault

```yaml
spring:
  cloud:
    vault:
      uri: https://vault.example.com
      authentication: TOKEN
      token: ${VAULT_TOKEN}
  config:
    import: vault://secret/spring-ai
```

### 多环境配置

```yaml
# application-dev.yml
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY_DEV}

# application-prod.yml
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY_PROD}
```

## 错误处理与重试

### 配置重试策略

```java
@Configuration
public class AiRetryConfig {

    @Bean
    public RetryTemplate aiRetryTemplate() {
        return RetryTemplate.builder()
                .maxAttempts(3)
                .exponentialBackoff(1000, 2, 10000)
                .retryOn(TransientAiException.class)
                .build();
    }
}
```

### 使用 Resilience4j

```java
@Service
public class ResilientChatService {

    private final ChatClient chatClient;

    @CircuitBreaker(name = "aiService", fallbackMethod = "fallbackChat")
    @RateLimiter(name = "aiService")
    @Retry(name = "aiService")
    public String chat(String message) {
        return chatClient.prompt().user(message).call().content();
    }

    public String fallbackChat(String message, Throwable t) {
        return "AI 服务暂时不可用，请稍后再试。错误: " + t.getMessage();
    }
}
```

```yaml
resilience4j:
  circuitbreaker:
    instances:
      aiService:
        failureRateThreshold: 50
        waitDurationInOpenState: 30s
        slidingWindowSize: 10
  ratelimiter:
    instances:
      aiService:
        limitForPeriod: 100
        limitRefreshPeriod: 1m
  retry:
    instances:
      aiService:
        maxAttempts: 3
        waitDuration: 1s
```

## 成本控制

### Token 使用监控

```java
@Aspect
@Component
public class TokenUsageAspect {

    private final MeterRegistry meterRegistry;

    @Around("execution(* org.springframework.ai.chat.ChatClient+.call(..))")
    public Object trackTokenUsage(ProceedingJoinPoint pjp) throws Throwable {
        Object result = pjp.proceed();

        if (result instanceof ChatResponse response) {
            Usage usage = response.getMetadata().getUsage();
            meterRegistry.counter("ai.tokens.prompt").increment(usage.getPromptTokens());
            meterRegistry.counter("ai.tokens.completion").increment(usage.getGenerationTokens());
            meterRegistry.counter("ai.tokens.total").increment(usage.getTotalTokens());
        }

        return result;
    }
}
```

### 设置 Token 限制

```java
@GetMapping("/chat")
public String chat(@RequestParam String message) {
    return chatClient.prompt()
            .user(message)
            .options(ChatOptionsBuilder.builder()
                    .withMaxTokens(500)  // 限制输出长度
                    .build())
            .call()
            .content();
}
```

### 缓存响应

```java
@Service
public class CachedChatService {

    private final ChatClient chatClient;
    private final Cache<String, String> cache;

    public CachedChatService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
        this.cache = Caffeine.newBuilder()
                .maximumSize(1000)
                .expireAfterWrite(Duration.ofHours(1))
                .build();
    }

    public String chat(String message) {
        String cacheKey = DigestUtils.md5Hex(message);
        return cache.get(cacheKey, key ->
                chatClient.prompt().user(message).call().content()
        );
    }
}
```

## 安全性

### 输入验证

```java
@Service
public class SecureChatService {

    private static final int MAX_MESSAGE_LENGTH = 4000;
    private static final Pattern INJECTION_PATTERN = Pattern.compile(
            "(?i)(ignore.*instructions|forget.*rules|system.*prompt)"
    );

    public String chat(String message) {
        // 长度检查
        if (message.length() > MAX_MESSAGE_LENGTH) {
            throw new IllegalArgumentException("消息过长");
        }

        // 注入检测
        if (INJECTION_PATTERN.matcher(message).find()) {
            throw new SecurityException("检测到潜在的 Prompt 注入");
        }

        // 敏感信息过滤
        String sanitized = sanitizeInput(message);

        return chatClient.prompt().user(sanitized).call().content();
    }

    private String sanitizeInput(String input) {
        // 移除或替换敏感信息模式
        return input.replaceAll("\\b\\d{16,19}\\b", "[CARD_NUMBER]")
                   .replaceAll("\\b\\d{3}-\\d{2}-\\d{4}\\b", "[SSN]");
    }
}
```

### 输出过滤

```java
@Service
public class OutputFilterService {

    private final List<String> sensitivePatterns = List.of(
            "password", "secret", "api_key", "token"
    );

    public String filterOutput(String aiResponse) {
        String filtered = aiResponse;
        for (String pattern : sensitivePatterns) {
            filtered = filtered.replaceAll(
                    "(?i)" + pattern + "\\s*[:=]\\s*\\S+",
                    pattern + ": [REDACTED]"
            );
        }
        return filtered;
    }
}
```

## 可观测性

### Micrometer 集成

```java
@Configuration
public class AiMetricsConfig {

    @Bean
    public ChatClientCustomizer metricsCustomizer(MeterRegistry registry) {
        return builder -> builder.defaultAdvisors(
                new MetricsAdvisor(registry)
        );
    }
}

public class MetricsAdvisor implements RequestResponseAdvisor {

    private final MeterRegistry registry;
    private final Timer.Builder timerBuilder;

    public MetricsAdvisor(MeterRegistry registry) {
        this.registry = registry;
        this.timerBuilder = Timer.builder("ai.chat.duration")
                .description("AI 请求耗时");
    }

    @Override
    public AdvisedRequest adviseRequest(AdvisedRequest request, Map<String, Object> context) {
        context.put("startTime", System.nanoTime());
        return request;
    }

    @Override
    public ChatResponse adviseResponse(ChatResponse response, Map<String, Object> context) {
        long startTime = (Long) context.get("startTime");
        long duration = System.nanoTime() - startTime;
        timerBuilder.register(registry).record(duration, TimeUnit.NANOSECONDS);
        return response;
    }
}
```

### 日志记录

```java
@Slf4j
@Aspect
@Component
public class AiLoggingAspect {

    @Around("execution(* org.springframework.ai.chat.ChatClient+.call(..))")
    public Object logAiCall(ProceedingJoinPoint pjp) throws Throwable {
        long start = System.currentTimeMillis();
        String requestId = UUID.randomUUID().toString().substring(0, 8);

        log.info("[{}] AI 请求开始", requestId);

        try {
            Object result = pjp.proceed();
            long duration = System.currentTimeMillis() - start;
            log.info("[{}] AI 请求完成, 耗时: {}ms", requestId, duration);
            return result;
        } catch (Exception e) {
            log.error("[{}] AI 请求失败: {}", requestId, e.getMessage());
            throw e;
        }
    }
}
```

## 测试策略

### Mock ChatClient

```java
@SpringBootTest
class ChatServiceTest {

    @MockBean
    private ChatClient chatClient;

    @Autowired
    private MyChatService chatService;

    @Test
    void testChat() {
        when(chatClient.prompt()).thenReturn(mockPromptBuilder());

        String result = chatService.processQuery("测试问题");

        assertThat(result).contains("预期关键词");
    }

    private ChatClient.PromptUserSpec mockPromptBuilder() {
        // 构建 mock 链
    }
}
```

### 使用 Testcontainers + Ollama

```java
@Testcontainers
@SpringBootTest
class IntegrationTest {

    @Container
    static GenericContainer<?> ollama = new GenericContainer<>("ollama/ollama:latest")
            .withExposedPorts(11434)
            .waitingFor(Wait.forHttp("/"));

    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.ai.ollama.base-url",
                () -> "http://localhost:" + ollama.getMappedPort(11434));
    }
}
```

## 下一步

- [FAQ](/docs/spring-ai/faq) - 常见问题解答
- [面试题](/docs/spring-ai/interview-questions) - Spring AI 面试题精选
