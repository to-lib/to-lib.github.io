---
sidebar_position: 15
title: 图像生成
---

# 图像生成

Spring AI 支持通过 `ImageClient` 接口调用图像生成模型，如 OpenAI DALL-E、Stability AI 等。

## ImageClient 基础

### 注入使用

```java
@RestController
public class ImageController {

    private final ImageClient imageClient;

    public ImageController(ImageClient imageClient) {
        this.imageClient = imageClient;
    }

    @PostMapping("/generate-image")
    public String generateImage(@RequestParam String description) {
        ImageResponse response = imageClient.call(
            new ImagePrompt(description)
        );
        return response.getResult().getOutput().getUrl();
    }
}
```

### 响应处理

```java
@PostMapping("/generate-image/full")
public Map<String, Object> generateImageFull(@RequestParam String prompt) {
    ImageResponse response = imageClient.call(new ImagePrompt(prompt));

    ImageGeneration result = response.getResult();
    Image image = result.getOutput();

    return Map.of(
        "url", image.getUrl(),
        "revisedPrompt", image.getRevisedPrompt(),  // 模型优化后的提示词
        "b64Json", image.getB64Json()  // Base64 编码（如果请求的话）
    );
}
```

## OpenAI DALL-E 配置

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
      image:
        options:
          model: dall-e-3 # 或 dall-e-2
          quality: hd # standard 或 hd
          size: 1024x1024 # 图像尺寸
          style: vivid # vivid 或 natural
          response-format: url # url 或 b64_json
```

### 模型对比

| 特性     | DALL-E 2           | DALL-E 3           |
| -------- | ------------------ | ------------------ |
| 质量     | 良好               | 最佳               |
| 尺寸选项 | 256, 512, 1024     | 1024, 1792x1024 等 |
| 提示优化 | 无                 | 自动优化提示词     |
| 成本     | 较低               | 较高               |
| 批量生成 | 支持（最多 10 张） | 每次 1 张          |

## ImageOptions 详解

### 编程式配置

```java
@PostMapping("/generate/custom")
public String customGenerate(@RequestBody ImageRequest request) {
    ImageOptions options = ImageOptionsBuilder.builder()
            .withModel("dall-e-3")
            .withQuality("hd")
            .withWidth(1792)
            .withHeight(1024)
            .withStyle("natural")
            .withN(1)  // 生成数量
            .build();

    ImagePrompt prompt = new ImagePrompt(request.description(), options);
    ImageResponse response = imageClient.call(prompt);

    return response.getResult().getOutput().getUrl();
}
```

### 尺寸选项

**DALL-E 3 支持的尺寸：**

- `1024x1024` - 正方形（默认）
- `1792x1024` - 横向/宽屏
- `1024x1792` - 纵向/竖屏

**DALL-E 2 支持的尺寸：**

- `256x256`
- `512x512`
- `1024x1024`

## 批量生成 (DALL-E 2)

```java
@PostMapping("/generate/batch")
public List<String> batchGenerate(@RequestParam String prompt, @RequestParam int count) {
    ImageOptions options = ImageOptionsBuilder.builder()
            .withModel("dall-e-2")
            .withN(Math.min(count, 10))  // 最多 10 张
            .build();

    ImagePrompt imagePrompt = new ImagePrompt(prompt, options);
    ImageResponse response = imageClient.call(imagePrompt);

    return response.getResults().stream()
            .map(gen -> gen.getOutput().getUrl())
            .collect(Collectors.toList());
}
```

## 获取 Base64 图像

```java
@PostMapping("/generate/base64")
public byte[] generateBase64(@RequestParam String prompt) {
    ImageOptions options = ImageOptionsBuilder.builder()
            .withResponseFormat("b64_json")
            .build();

    ImagePrompt imagePrompt = new ImagePrompt(prompt, options);
    ImageResponse response = imageClient.call(imagePrompt);

    String b64 = response.getResult().getOutput().getB64Json();
    return Base64.getDecoder().decode(b64);
}
```

## 提示词工程

### 有效的图像提示词

```java
public String buildImagePrompt(ImageGenerationRequest request) {
    StringBuilder prompt = new StringBuilder();

    // 主题
    prompt.append(request.subject());

    // 风格
    if (request.style() != null) {
        prompt.append(", in the style of ").append(request.style());
    }

    // 氛围
    if (request.mood() != null) {
        prompt.append(", with a ").append(request.mood()).append(" atmosphere");
    }

    // 细节
    if (request.details() != null) {
        prompt.append(", ").append(request.details());
    }

    // 质量关键词
    prompt.append(", highly detailed, professional quality, 4k");

    return prompt.toString();
}
```

### 示例提示词

| 场景     | 提示词示例                                                                   |
| -------- | ---------------------------------------------------------------------------- |
| 产品图   | "Professional product photo of [product], studio lighting, white background" |
| 插图     | "Digital illustration of [subject], flat design, vibrant colors"             |
| 照片风格 | "Photorealistic image of [subject], natural lighting, DSLR quality"          |
| 艺术作品 | "Oil painting of [subject], impressionist style, rich textures"              |

## 错误处理

```java
@PostMapping("/generate/safe")
public ResponseEntity<?> safeGenerate(@RequestParam String prompt) {
    try {
        ImageResponse response = imageClient.call(new ImagePrompt(prompt));
        return ResponseEntity.ok(Map.of(
            "url", response.getResult().getOutput().getUrl()
        ));
    } catch (ContentPolicyViolationException e) {
        return ResponseEntity.badRequest().body(Map.of(
            "error", "内容违反使用政策",
            "message", "请修改描述后重试"
        ));
    } catch (RateLimitException e) {
        return ResponseEntity.status(429).body(Map.of(
            "error", "请求过于频繁",
            "retryAfter", e.getRetryAfter()
        ));
    } catch (Exception e) {
        return ResponseEntity.internalServerError().body(Map.of(
            "error", "图像生成失败"
        ));
    }
}
```

## 成本优化

### 1. 缓存生成结果

```java
@Service
public class CachedImageService {

    private final ImageClient imageClient;
    private final Cache<String, String> cache;

    public String generateWithCache(String prompt) {
        String key = DigestUtils.md5Hex(prompt);
        return cache.get(key, k -> {
            ImageResponse response = imageClient.call(new ImagePrompt(prompt));
            return response.getResult().getOutput().getUrl();
        });
    }
}
```

### 2. 选择合适的质量

```java
// 预览/草图：使用 DALL-E 2 + 较小尺寸
ImageOptions previewOptions = ImageOptionsBuilder.builder()
        .withModel("dall-e-2")
        .withSize("256x256")
        .build();

// 最终产品：使用 DALL-E 3 + HD
ImageOptions finalOptions = ImageOptionsBuilder.builder()
        .withModel("dall-e-3")
        .withQuality("hd")
        .withSize("1024x1024")
        .build();
```

## 支持的提供商

| 提供商       | 模型             | 说明         |
| ------------ | ---------------- | ------------ |
| OpenAI       | DALL-E 2/3       | 最成熟的选择 |
| Stability AI | Stable Diffusion | 开源模型生态 |
| Azure OpenAI | DALL-E           | 企业级合规   |

## 下一步

- [模型提供商](/docs/spring-ai/model-providers) - 了解所有支持的提供商
- [最佳实践](/docs/spring-ai/best-practices) - 生产环境最佳实践
