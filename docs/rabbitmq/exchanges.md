---
sidebar_position: 4
title: "交换机详解"
description: "RabbitMQ Exchange 类型、路由规则与常见用法"
---

# 交换机详解

交换机（Exchange）负责接收生产者发布的消息，并根据**交换机类型**与**绑定（Binding）规则**把消息路由到一个或多个队列。

## 交换机的核心属性

- **name**：交换机名称（同 vhost 内唯一）
- **type**：交换机类型（direct/fanout/topic/headers 等）
- **durable**：是否持久化（Broker 重启后是否保留）
- **autoDelete**：当最后一个绑定被删除后是否自动删除
- **internal**：是否为内部交换机（只能被其他交换机绑定/路由，不能由 Producer 直接 publish）

Java 声明示例：

```java
channel.exchangeDeclare("order.exchange", "direct", true);
```

Spring（声明式）示例：

```java
@Bean
public DirectExchange orderExchange() {
    return ExchangeBuilder.directExchange("order.exchange").durable(true).build();
}
```

## 交换机类型

## Direct Exchange（直连）

按 routing key **完全匹配**路由。

- **适用场景**
- **[点对点]**：一个 routing key 对应一类队列
- **[多路分发]**：同一个 routing key 绑定多个队列，可实现多消费者组

```java
channel.exchangeDeclare("direct.logs", "direct", true);
channel.queueDeclare("error.queue", true, false, false, null);
channel.queueBind("error.queue", "direct.logs", "error");

channel.basicPublish("direct.logs", "error", null, "E".getBytes());
```

## Fanout Exchange（扇出）

忽略 routing key，**广播**到所有绑定的队列。

- **适用场景**
- **[广播通知]**：缓存失效通知、配置变更通知
- **[日志收集]**：多个日志处理服务都要收到

```java
channel.exchangeDeclare("broadcast", "fanout", true);
channel.basicPublish("broadcast", "", null, "hello".getBytes());
```

## Topic Exchange（主题）

按 routing key **模式匹配**路由。

- `*`：匹配一个单词
- `#`：匹配零个或多个单词

```java
channel.exchangeDeclare("topic.logs", "topic", true);
channel.queueDeclare("kern.queue", true, false, false, null);
channel.queueBind("kern.queue", "topic.logs", "kern.*");
channel.basicPublish("topic.logs", "kern.critical", null, "C".getBytes());
```

## Headers Exchange（头交换机）

不依赖 routing key，而是基于 headers 匹配（常见策略：`x-match=all|any`）。

```java
channel.exchangeDeclare("headers.ex", "headers", true);

Map<String, Object> bindHeaders = new HashMap<>();
bindHeaders.put("x-match", "all");
bindHeaders.put("format", "pdf");
channel.queueBind("pdf.queue", "headers.ex", "", bindHeaders);
```

## 常见但容易忽略的交换机

## Default Exchange（默认交换机，名字为空字符串）

- 交换机名为 `""`
- 规则：routing key 直接当作队列名

```java
channel.queueDeclare("q1", true, false, false, null);
channel.basicPublish("", "q1", null, "msg".getBytes());
```

## Alternate Exchange（备用交换机，处理无法路由的消息）

当消息无法路由到任何队列时：

- **方案 A**：Producer 设置 `mandatory=true` + ReturnListener 接回
- **方案 B**：为交换机配置 `alternate-exchange`，把 unroutable 消息路由到备用交换机（推荐做法之一）

声明示例：

```java
Map<String, Object> args = new HashMap<>();
args.put("alternate-exchange", "ae.exchange");
channel.exchangeDeclare("main.exchange", "direct", true, false, args);

channel.exchangeDeclare("ae.exchange", "fanout", true);
channel.queueDeclare("unroutable.queue", true, false, false, null);
channel.queueBind("unroutable.queue", "ae.exchange", "");
```

## Exchange-to-Exchange Binding（交换机绑定交换机）

适用于搭建更复杂的路由拓扑（注意避免循环）。

```java
channel.exchangeDeclare("source.ex", "topic", true);
channel.exchangeDeclare("dest.ex", "topic", true);
channel.exchangeBind("dest.ex", "source.ex", "order.#");
```

## 与可靠性相关的要点

- **durable exchange + durable queue + persistent message** 才构成“重启后仍保留”的基础
- **Publisher Confirm** 解决“生产者不知道消息是否到达 Broker”的问题（见 `producer` 与 `message-types`）
- **mandatory + return** 或 **alternate exchange** 解决“到达交换机但无法路由到队列”的问题

## 下一步

- 📊 [队列管理](/docs/rabbitmq/queues) - 了解队列属性、TTL、DLX、Quorum/Stream
- 💻 [生产者指南](/docs/rabbitmq/producer) - 生产者确认、return、序列化与重试
- 🎯 [核心概念](/docs/rabbitmq/core-concepts) - Connection/Channel/Binding/Queue 等基础
