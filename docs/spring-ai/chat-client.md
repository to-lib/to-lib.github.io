---
sidebar_position: 5
title: ChatClient 详解
---

# ChatClient 详解

`ChatClient` 是 Spring AI 中最核心的组件，用于与各种聊天模型进行交互。

## 基本用法

### 简单调用

```java
@RestController
public class ChatController {

    private final ChatClient chatClient;

    public ChatController(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder.build();
    }

    @GetMapping("/chat")
    public String chat(@RequestParam String message) {
        return chatClient.prompt()
                .user(message)
                .call()
                .content();
    }
}
```

### 使用 Prompt 对象

```java
@GetMapping("/chat/prompt")
public String chatWithPrompt(@RequestParam String message) {
    Prompt prompt = new Prompt(new UserMessage(message));
    ChatResponse response = chatClient.prompt(prompt).call().chatResponse();
    return response.getResult().getOutput().getContent();
}
```

## 流式响应

使用 `stream()` 方法获取流式响应，适用于需要实时显示输出的场景：

```java
@GetMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public Flux<String> chatStream(@RequestParam String message) {
    return chatClient.prompt()
            .user(message)
            .stream()
            .content();
}
```

## 多轮对话

通过维护消息历史实现多轮对话：

```java
@RestController
public class ConversationController {

    private final ChatClient chatClient;
    private final List<Message> conversationHistory = new ArrayList<>();

    public ConversationController(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    @PostMapping("/conversation")
    public String converse(@RequestBody String userMessage) {
        conversationHistory.add(new UserMessage(userMessage));

        ChatResponse response = chatClient.prompt()
                .messages(conversationHistory)
                .call()
                .chatResponse();

        String assistantMessage = response.getResult().getOutput().getContent();
        conversationHistory.add(new AssistantMessage(assistantMessage));

        return assistantMessage;
    }

    @DeleteMapping("/conversation")
    public void clearHistory() {
        conversationHistory.clear();
    }
}
```

## 系统提示词

使用系统消息设置 AI 的行为和角色：

```java
@GetMapping("/chat/system")
public String chatWithSystem(@RequestParam String message) {
    return chatClient.prompt()
            .system("你是一个专业的 Java 开发助手，请用中文回答问题。")
            .user(message)
            .call()
            .content();
}
```

## Function Calling (函数调用)

Spring AI 支持让 AI 模型调用预定义的函数：

### 定义函数

```java
@Configuration
public class FunctionConfig {

    @Bean
    @Description("获取指定城市的当前天气")
    public Function<WeatherRequest, WeatherResponse> weatherFunction() {
        return request -> {
            // 实际调用天气 API
            return new WeatherResponse(request.city(), "晴天", 25);
        };
    }

    public record WeatherRequest(String city) {}
    public record WeatherResponse(String city, String condition, int temperature) {}
}
```

### 使用函数

```java
@GetMapping("/chat/function")
public String chatWithFunction(@RequestParam String message) {
    return chatClient.prompt()
            .user(message)
            .functions("weatherFunction")
            .call()
            .content();
}
```

当用户询问"北京今天天气怎么样？"时，AI 会自动调用 `weatherFunction` 获取数据。

## ChatOptions 配置

自定义模型参数：

```java
@GetMapping("/chat/options")
public String chatWithOptions(@RequestParam String message) {
    return chatClient.prompt()
            .user(message)
            .options(ChatOptionsBuilder.builder()
                    .withTemperature(0.7f)  // 创造性程度
                    .withMaxTokens(500)     // 最大输出长度
                    .withTopP(0.9f)         // 核采样
                    .build())
            .call()
            .content();
}
```

### 常用参数说明

| 参数          | 说明                                         | 默认值     |
| ------------- | -------------------------------------------- | ---------- |
| `temperature` | 控制输出的随机性，0-1 之间，值越高越有创造性 | 0.7        |
| `maxTokens`   | 生成的最大 token 数量                        | 模型默认值 |
| `topP`        | 核采样，控制输出的多样性                     | 1.0        |
| `topK`        | 限制每步选择的候选词数量                     | 无         |

## 错误处理

```java
@GetMapping("/chat/safe")
public ResponseEntity<String> safeCaht(@RequestParam String message) {
    try {
        String response = chatClient.prompt()
                .user(message)
                .call()
                .content();
        return ResponseEntity.ok(response);
    } catch (NonTransientAiException e) {
        // 不可重试的错误（如 API key 无效）
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                .body("配置错误: " + e.getMessage());
    } catch (TransientAiException e) {
        // 可重试的错误（如网络问题、限流）
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body("服务暂时不可用，请稍后重试");
    }
}
```

## 下一步

- [Prompt 工程](/docs/spring-ai/prompts) - 学习如何编写有效的提示词
- [输出解析](/docs/spring-ai/output-parsing) - 将 AI 响应转换为结构化数据
