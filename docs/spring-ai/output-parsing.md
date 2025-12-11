---
sidebar_position: 7
title: 输出解析
---

# 输出解析

AI 模型通常返回非结构化的文本。Spring AI 的输出解析器（Output Parser）可以将这些文本转换为 Java 对象（POJO），便于程序处理。

## BeanOutputParser

将 AI 输出解析为 Java Bean：

```java
public record BookRecommendation(
    String title,
    String author,
    String genre,
    String summary
) {}

@GetMapping("/recommend")
public BookRecommendation recommendBook(@RequestParam String topic) {
    BeanOutputParser<BookRecommendation> parser = new BeanOutputParser<>(BookRecommendation.class);

    String userMessage = String.format("""
        推荐一本关于 %s 的书籍。
        %s
        """, topic, parser.getFormat());

    String response = chatClient.prompt()
            .user(userMessage)
            .call()
            .content();

    return parser.parse(response);
}
```

`parser.getFormat()` 会生成 JSON Schema 格式的说明，告诉 AI 如何格式化输出。

## ListOutputParser

解析列表输出：

```java
@GetMapping("/ideas")
public List<String> generateIdeas(@RequestParam String topic) {
    ListOutputParser parser = new ListOutputParser(new DefaultConversionService());

    String userMessage = String.format("""
        列出 5 个关于 %s 的创意点子。
        %s
        """, topic, parser.getFormat());

    String response = chatClient.prompt()
            .user(userMessage)
            .call()
            .content();

    return parser.parse(response);
}
```

## MapOutputParser

解析键值对输出：

```java
@GetMapping("/analyze")
public Map<String, Object> analyzeText(@RequestParam String text) {
    MapOutputParser parser = new MapOutputParser();

    String userMessage = String.format("""
        分析以下文本并提供：
        - sentiment: 情感倾向 (positive/negative/neutral)
        - keywords: 关键词列表
        - summary: 一句话摘要

        文本：%s

        %s
        """, text, parser.getFormat());

    String response = chatClient.prompt()
            .user(userMessage)
            .call()
            .content();

    return parser.parse(response);
}
```

## 复杂对象解析

### 嵌套对象

```java
public record TechArticle(
    String title,
    Author author,
    List<String> tags,
    Content content
) {}

public record Author(String name, String email) {}

public record Content(String introduction, List<Section> sections) {}

public record Section(String heading, String body) {}

@GetMapping("/article")
public TechArticle generateArticle(@RequestParam String topic) {
    BeanOutputParser<TechArticle> parser = new BeanOutputParser<>(TechArticle.class);

    String response = chatClient.prompt()
            .system("你是一个技术博客作者")
            .user("写一篇关于 " + topic + " 的技术文章。" + parser.getFormat())
            .call()
            .content();

    return parser.parse(response);
}
```

### 使用 Jackson 注解

```java
public record ApiResponse(
    @JsonProperty("status_code") int statusCode,
    @JsonProperty("error_message") String errorMessage,
    @JsonProperty("data") ResponseData data
) {}
```

## 使用 Converter

Spring AI 集成了 Spring 的 ConversionService：

```java
@Configuration
public class ConverterConfig {

    @Bean
    public ConversionService conversionService() {
        DefaultConversionService service = new DefaultConversionService();
        service.addConverter(new StringToLocalDateConverter());
        return service;
    }
}

public class StringToLocalDateConverter implements Converter<String, LocalDate> {
    @Override
    public LocalDate convert(String source) {
        return LocalDate.parse(source, DateTimeFormatter.ISO_DATE);
    }
}
```

## 错误处理

解析可能失败，需要妥善处理：

```java
@GetMapping("/parse-safe")
public ResponseEntity<?> safeParseExample(@RequestParam String input) {
    BeanOutputParser<MyDto> parser = new BeanOutputParser<>(MyDto.class);

    try {
        String response = chatClient.prompt()
                .user(input + "\n" + parser.getFormat())
                .call()
                .content();

        MyDto result = parser.parse(response);
        return ResponseEntity.ok(result);

    } catch (JsonParseException e) {
        // AI 返回的不是有效 JSON
        return ResponseEntity.badRequest()
                .body(Map.of("error", "AI 响应格式错误", "details", e.getMessage()));
    } catch (Exception e) {
        return ResponseEntity.internalServerError()
                .body(Map.of("error", "解析失败", "details", e.getMessage()));
    }
}
```

## 重试机制

当解析失败时自动重试：

```java
@Service
public class RobustParsingService {

    private final ChatClient chatClient;

    @Retryable(
        value = JsonParseException.class,
        maxAttempts = 3,
        backoff = @Backoff(delay = 1000)
    )
    public <T> T parseWithRetry(String prompt, Class<T> targetClass) {
        BeanOutputParser<T> parser = new BeanOutputParser<>(targetClass);

        String response = chatClient.prompt()
                .user(prompt + "\n请严格按照以下 JSON 格式输出：\n" + parser.getFormat())
                .call()
                .content();

        return parser.parse(response);
    }
}
```

## 自定义 OutputParser

实现 `OutputParser` 接口创建自定义解析器：

```java
public class MarkdownTableParser implements OutputParser<List<Map<String, String>>> {

    @Override
    public List<Map<String, String>> parse(String text) {
        List<Map<String, String>> result = new ArrayList<>();
        String[] lines = text.split("\n");
        String[] headers = null;

        for (String line : lines) {
            if (line.startsWith("|") && line.endsWith("|")) {
                String[] cells = line.substring(1, line.length() - 1).split("\\|");
                if (headers == null) {
                    headers = Arrays.stream(cells).map(String::trim).toArray(String[]::new);
                } else if (!line.contains("---")) {
                    Map<String, String> row = new HashMap<>();
                    for (int i = 0; i < headers.length && i < cells.length; i++) {
                        row.put(headers[i], cells[i].trim());
                    }
                    result.add(row);
                }
            }
        }
        return result;
    }

    @Override
    public String getFormat() {
        return "请以 Markdown 表格格式输出，包含表头和数据行。";
    }
}
```

## 下一步

- [RAG 应用](./rag) - 构建检索增强生成应用
- [ChatClient 详解](./chat-client) - 深入了解 ChatClient 的高级用法
