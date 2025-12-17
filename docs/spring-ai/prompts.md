---
sidebar_position: 6
title: Prompt 工程
---

# Prompt 工程

Prompt（提示词）是与 AI 模型交互的核心。优质的 Prompt 能够显著提升 AI 输出的质量和准确性。

## PromptTemplate

`PromptTemplate` 允许创建带有占位符的可复用提示词模板：

```java
@Service
public class PromptService {

    public Prompt createJokePrompt(String topic, String style) {
        PromptTemplate template = new PromptTemplate(
            "请讲一个关于{topic}的{style}笑话"
        );
        return template.create(Map.of(
            "topic", topic,
            "style", style
        ));
    }
}
```

### 从资源文件加载模板

将 Prompt 模板存储在外部文件中便于管理：

```java
// src/main/resources/prompts/analysis.st
@Component
public class AnalysisPromptService {

    @Value("classpath:prompts/analysis.st")
    private Resource analysisPrompt;

    public Prompt createAnalysisPrompt(String code) {
        PromptTemplate template = new PromptTemplate(analysisPrompt);
        return template.create(Map.of("code", code));
    }
}
```

**prompts/analysis.st:**

````text
请分析以下代码并提供改进建议：

```java
{code}
```

请从以下几个方面进行分析：

1. 代码质量
2. 性能问题
3. 安全隐患
4. 最佳实践
````

## 消息类型

Spring AI 支持多种消息类型，用于构建复杂的对话：

### SystemMessage

设置 AI 的行为、角色和约束：

```java
Message systemMessage = new SystemMessage("""
    你是一个专业的技术文档撰写助手。
    请遵循以下规则：
    1. 使用清晰、简洁的语言
    2. 提供代码示例时使用 Markdown 格式
    3. 如果不确定，请明确说明
    """);
```

### UserMessage

用户输入：

```java
Message userMessage = new UserMessage("解释什么是 Spring Boot 自动配置");
```

### AssistantMessage

AI 的历史回复（用于多轮对话）：

```java
Message assistantMessage = new AssistantMessage("Spring Boot 自动配置是...");
```

### 组合使用

```java
List<Message> messages = List.of(
    new SystemMessage("你是一个 Java 专家"),
    new UserMessage("什么是依赖注入？"),
    new AssistantMessage("依赖注入是一种设计模式..."),
    new UserMessage("能给我一个例子吗？")
);

ChatResponse response = chatClient.prompt()
        .messages(messages)
        .call()
        .chatResponse();
```

## Prompt 最佳实践

### 1. 明确角色和上下文

```java
String systemPrompt = """
    角色：你是一位资深的 Spring 框架专家，拥有 10 年以上的 Java 开发经验。
    任务：回答用户关于 Spring 生态系统的技术问题。
    约束：
    - 使用中文回答
    - 提供代码示例时使用 Java 17+ 的语法
    - 如果问题超出 Spring 范围，礼貌地说明并提供相关资源
    """;
```

### 2. 使用结构化输出格式

```java
String userPrompt = """
    分析以下异常信息并提供解决方案：

    {exception}

    请按以下格式回答：
    ## 原因分析
    [异常发生的根本原因]

    ## 解决方案
    [具体的解决步骤]

    ## 预防措施
    [如何避免类似问题]
    """;
```

### 3. Few-shot Learning

提供示例帮助 AI 理解期望的输出格式：

```java
String prompt = """
    将以下英文技术术语翻译为中文，保持专业性：

    示例：
    - Dependency Injection -> 依赖注入
    - Aspect-Oriented Programming -> 面向切面编程
    - Inversion of Control -> 控制反转

    请翻译：
    - {term}
    """;
```

### 4. Chain of Thought (思维链)

引导 AI 逐步推理：

````java
String prompt = """
    请分析以下代码的时间复杂度。

    ```java
    {code}
    ```

    请按以下步骤分析：
    1. 首先，识别代码中的循环结构
    2. 然后，分析每个循环的迭代次数
    3. 接着，考虑嵌套循环的影响
    4. 最后，给出最终的时间复杂度，并解释原因
    """;
````

## 动态 Prompt 构建

使用构建器模式动态构建复杂 Prompt：

````java
@Service
public class DynamicPromptService {

    public Prompt buildCodeReviewPrompt(String code, List<String> focusAreas) {
        StringBuilder sb = new StringBuilder();
        sb.append("请对以下代码进行审查：\n\n```java\n");
        sb.append(code);
        sb.append("\n```\n\n");
        sb.append("重点关注以下方面：\n");

        for (int i = 0; i < focusAreas.size(); i++) {
            sb.append(String.format("%d. %s\n", i + 1, focusAreas.get(i)));
        }

        return new Prompt(new UserMessage(sb.toString()));
    }
}
````

## 下一步

- [输出解析](/docs/spring-ai/output-parsing) - 将 AI 响应转换为结构化数据
- [RAG 应用](/docs/spring-ai/rag) - 构建检索增强生成应用
