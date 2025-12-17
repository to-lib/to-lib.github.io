---
sidebar_position: 18
title: 对话记忆（ChatMemory）
---

# 对话记忆（ChatMemory）

在实际业务中，用户往往会进行多轮对话：

- 上一句的指代（“它”“这个方案”）需要上下文才能理解
- 需要记住用户偏好（语言、格式、领域、身份信息等）
- 需要把历史消息注入到模型上下文中，形成“连续对话”体验

Spring AI 用 `ChatMemory` 抽象来管理对话历史，并通过 `MessageChatMemoryAdvisor` 将记忆无侵入地接入 `ChatClient`。

## 核心概念

- **ChatMemory**：对话记忆的存储与检索接口
- **conversationId**：会话 ID（同一用户、同一会话的唯一标识）
- **lastN / window**：仅取最近 N 条消息（避免上下文过长导致成本飙升）
- **Advisor**：把“读写记忆”的逻辑挂到请求前/响应后处理链

## ChatMemory 接口

Spring AI 提供 `ChatMemory` 接口用于：

- 写入新消息
- 读取最近 N 条历史
- 清空会话

```java
public interface ChatMemory {
    void add(String conversationId, List<Message> messages);
    List<Message> get(String conversationId, int lastN);
    void clear(String conversationId);
}
```

## 使用 MessageChatMemoryAdvisor

最常见的用法是：

1. 选择一个 `ChatMemory` 实现（例如 `InMemoryChatMemory`）
2. 将 `MessageChatMemoryAdvisor` 作为默认 Advisor
3. 在每次请求时传入 `conversationId`

```java
ChatClient client = ChatClient.builder(chatModel)
    .defaultAdvisors(new MessageChatMemoryAdvisor(new InMemoryChatMemory()))
    .build();

String content = client.prompt()
    .user("你好，我叫张三")
    .advisors(a -> a.param(MessageChatMemoryAdvisor.CHAT_MEMORY_CONVERSATION_ID_KEY, "u-1001"))
    .call()
    .content();
```

后续同一 `conversationId` 的请求将自动带上历史消息：

```java
String content = client.prompt()
    .user("我刚才叫什么名字？")
    .advisors(a -> a.param(MessageChatMemoryAdvisor.CHAT_MEMORY_CONVERSATION_ID_KEY, "u-1001"))
    .call()
    .content();
```

## 记忆窗口（只取最近 N 条）

对话历史越长：

- 请求 token 越大，成本越高
- 延迟越高
- 可能触发模型上下文窗口限制

建议：

- 对于客服问答：通常 `lastN = 10~30` 足够
- 对于复杂任务/长链路：结合摘要（summary）或“关键事实记忆”

如果你需要更精细的窗口控制，可以：

- 保留最近 N 条原始消息
- 更早的消息做一次摘要（作为 system message 或 memory message）

## 持久化与多实例部署

`InMemoryChatMemory` 适合开发和单实例。

当你在生产环境：

- 多实例部署（K8s 横向扩容）
- 需要跨重启保留会话

就应考虑实现一个可持久化的 `ChatMemory`（例如基于 Redis / 数据库），并处理：

- **TTL**：会话过期与清理
- **容量控制**：限制每个会话的最大消息数
- **敏感信息**：入库前脱敏/加密

## 最佳实践

- **强制传入会话 ID**：不要把所有用户的对话都写到同一个 `conversationId`
- **限制历史长度**：避免无限增长
- **分层记忆**：短期记忆（最近对话）+ 长期记忆（用户画像/偏好）
- **敏感信息处理**：避免把密码、身份证、API Key 等写入记忆

## 下一步

- [Advisor 机制](/docs/spring-ai/advisors)
- [RAG 应用](/docs/spring-ai/rag)
- [最佳实践](/docs/spring-ai/best-practices)
