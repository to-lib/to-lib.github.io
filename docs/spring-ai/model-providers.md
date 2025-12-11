---
sidebar_position: 9
title: 模型提供商
---

# 模型提供商

Spring AI 支持多种 AI 模型提供商，并提供统一的 API 接口，使切换提供商变得简单。

## 支持的提供商

| 提供商           | 聊天模型 | 嵌入模型 | 图像模型 | 说明                   |
| ---------------- | :------: | :------: | :------: | ---------------------- |
| OpenAI           |    ✅    |    ✅    |    ✅    | GPT-4, GPT-3.5, DALL-E |
| Azure OpenAI     |    ✅    |    ✅    |    ✅    | 企业级 OpenAI          |
| Anthropic        |    ✅    |    ❌    |    ❌    | Claude 系列            |
| Ollama           |    ✅    |    ✅    |    ❌    | 本地部署开源模型       |
| Amazon Bedrock   |    ✅    |    ✅    |    ✅    | AWS 托管多模型         |
| Google Vertex AI |    ✅    |    ✅    |    ✅    | Gemini, PaLM           |
| Hugging Face     |    ✅    |    ✅    |    ❌    | 开源模型生态           |
| Mistral AI       |    ✅    |    ✅    |    ❌    | Mistral 系列           |

## OpenAI

### 依赖

```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
```

### 配置

```yaml
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
      chat:
        options:
          model: gpt-4
          temperature: 0.7
      embedding:
        options:
          model: text-embedding-ada-002
      image:
        options:
          model: dall-e-3
```

### 使用

```java
@RestController
public class OpenAiController {

    private final ChatClient chatClient;
    private final ImageClient imageClient;

    @GetMapping("/openai/chat")
    public String chat(@RequestParam String message) {
        return chatClient.prompt().user(message).call().content();
    }

    @GetMapping("/openai/image")
    public String generateImage(@RequestParam String description) {
        ImageResponse response = imageClient.call(
            new ImagePrompt(description)
        );
        return response.getResult().getOutput().getUrl();
    }
}
```

## Azure OpenAI

企业级 OpenAI 服务，适合有合规要求的场景。

### 依赖

```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-azure-openai-spring-boot-starter</artifactId>
</dependency>
```

### 配置

```yaml
spring:
  ai:
    azure:
      openai:
        api-key: ${AZURE_OPENAI_API_KEY}
        endpoint: https://your-resource.openai.azure.com/
        deployment-name: gpt-4-deployment
```

## Ollama (本地部署)

在本地运行开源大模型，无需 API 费用，数据完全私有。

### 安装 Ollama

```bash
# macOS
brew install ollama

# 启动服务
ollama serve

# 拉取模型
ollama pull llama2
ollama pull mistral
ollama pull codellama
```

### 依赖

```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-ollama-spring-boot-starter</artifactId>
</dependency>
```

### 配置

```yaml
spring:
  ai:
    ollama:
      base-url: http://localhost:11434
      chat:
        options:
          model: llama2
          temperature: 0.7
      embedding:
        options:
          model: nomic-embed-text
```

### 推荐模型

| 用途     | 模型                          | 大小   |
| -------- | ----------------------------- | ------ |
| 通用对话 | `llama2`, `mistral`           | 7B     |
| 代码生成 | `codellama`, `deepseek-coder` | 7B-34B |
| 中文对话 | `qwen`, `yi`                  | 7B-34B |
| 轻量级   | `phi`, `gemma:2b`             | 2B-3B  |

## Anthropic Claude

以安全和长上下文著称。

### 依赖

```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-anthropic-spring-boot-starter</artifactId>
</dependency>
```

### 配置

```yaml
spring:
  ai:
    anthropic:
      api-key: ${ANTHROPIC_API_KEY}
      chat:
        options:
          model: claude-3-opus-20240229
          max-tokens: 4096
```

## Amazon Bedrock

AWS 托管的多模型服务。

### 依赖

```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-bedrock-ai-spring-boot-starter</artifactId>
</dependency>
```

### 配置

```yaml
spring:
  ai:
    bedrock:
      aws:
        region: us-east-1
        access-key: ${AWS_ACCESS_KEY}
        secret-key: ${AWS_SECRET_KEY}
      titan:
        chat:
          enabled: true
          model: amazon.titan-text-express-v1
```

## 动态切换提供商

### 多 ChatClient 配置

```java
@Configuration
public class MultiModelConfig {

    @Bean("openaiClient")
    public ChatClient openaiClient(OpenAiChatClient openAiChatClient) {
        return ChatClient.builder(openAiChatClient).build();
    }

    @Bean("ollamaClient")
    public ChatClient ollamaClient(OllamaChatClient ollamaChatClient) {
        return ChatClient.builder(ollamaChatClient).build();
    }
}
```

### 使用

```java
@RestController
public class MultiModelController {

    @Qualifier("openaiClient")
    private final ChatClient openaiClient;

    @Qualifier("ollamaClient")
    private final ChatClient ollamaClient;

    @GetMapping("/chat")
    public String chat(
            @RequestParam String message,
            @RequestParam(defaultValue = "openai") String provider) {

        ChatClient client = switch (provider) {
            case "ollama" -> ollamaClient;
            default -> openaiClient;
        };

        return client.prompt().user(message).call().content();
    }
}
```

## 模型选择建议

| 场景     | 推荐模型         | 原因                 |
| -------- | ---------------- | -------------------- |
| 生产环境 | GPT-4, Claude 3  | 能力强，稳定可靠     |
| 成本敏感 | GPT-3.5, Mistral | 性价比高             |
| 数据隐私 | Ollama + Llama2  | 完全本地，无数据外泄 |
| 长文档   | Claude 3         | 支持 200K 上下文     |
| 代码生成 | GPT-4, CodeLlama | 代码能力强           |

## 下一步

- [最佳实践](/docs/spring-ai/best-practices) - 生产环境配置与优化
- [FAQ](/docs/spring-ai/faq) - 常见问题解答
