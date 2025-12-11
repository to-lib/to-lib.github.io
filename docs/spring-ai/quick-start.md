---
sidebar_position: 2
title: 快速开始
---

# 快速开始

本指南将带你快速构建一个基于 Spring AI 和 OpenAI 的简单应用。

## 前置条件

- JDK 17+
- Maven 3.x 或 Gradle
- OpenAI API Key

## 1. 创建项目

你可以使用 [Spring Initializr](https://start.spring.io/) 创建一个新项目，添加 `Spring Web` 和 `Spring AI OpenAI` 依赖。

或者在 `pom.xml` 中手动添加依赖：

```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
```

注意：Spring AI 目前处于 Milestone 或 Snapshot 阶段，你可能需要配置 Spring 仓库。

## 2. 配置 API Key

在 `application.properties` 或 `application.yml` 中配置你的 OpenAI API Key：

```yaml
spring:
  ai:
    openai:
      api-key: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 3. 编写代码

创建一个简单的 Controller 来调用 AI 模型：

```java
package com.example.demo;

import org.springframework.ai.chat.ChatClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ChatController {

    private final ChatClient chatClient;

    public ChatController(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    @GetMapping("/chat")
    public String chat(@RequestParam(value = "message", defaultValue = "Tell me a joke") String message) {
        return chatClient.call(message);
    }
}
```

## 4. 运行应用

启动 Spring Boot 应用，并访问：

```
http://localhost:8080/chat?message=Hello
```

你应该能看到来自 OpenAI 的回复。

## 5. 总结

你已经成功集成了一个基本的 AI 聊天功能。Spring AI 自动配置了 `ChatClient`，你只需要注入并使用它。
