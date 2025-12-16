---
sidebar_position: 5
title: "队列管理"
description: "RabbitMQ Queue 参数、TTL、DLX、队列类型与实践"
---

# 队列管理

队列（Queue）是消息最终存储与消费的位置。队列的配置决定了：消息是否持久化、是否会过期、是否会进入死信、队列满了如何处理、以及队列高可用方案等。

## 队列的基础属性

- **name**：队列名
- **durable**：是否持久化（Broker 重启后仍存在）
- **exclusive**：排他队列（仅当前连接可用，连接断开即删除）
- **autoDelete**：最后一个消费者取消订阅后自动删除

Java 声明示例：

```java
channel.queueDeclare("order.queue", true, false, false, null);
```

## 常用队列 arguments（核心参数）

通过 `arguments` 扩展队列行为。

```java
Map<String, Object> args = new HashMap<>();
args.put("x-message-ttl", 60000);
args.put("x-dead-letter-exchange", "dlx.exchange");
args.put("x-dead-letter-routing-key", "dlx.order");
channel.queueDeclare("order.queue", true, false, false, args);
```

## TTL（过期）

RabbitMQ 支持：

- **消息 TTL**：每条消息的过期时间（message property `expiration`）
- **队列 TTL**：队列级别对所有消息统一设置（`x-message-ttl`）
- **队列过期**：队列无人使用时自动过期（`x-expires`）

## 消息 TTL（per-message）

```java
AMQP.BasicProperties props = new AMQP.BasicProperties.Builder()
    .expiration("30000")
    .build();
channel.basicPublish("", "q", props, body);
```

## 队列 TTL（per-queue）

```java
args.put("x-message-ttl", 30000);
```

:::warning 注意
消息 TTL 的过期检查以队列头部为主：如果队头消息 TTL 更长，后续短 TTL 消息可能无法及时过期处理。
:::

## DLX（Dead Letter Exchange，死信交换机）

消息进入死信的典型情况：

- 被 `basicReject/basicNack` 且 `requeue=false`
- 消息过期（TTL）
- 队列达到最大长度（`x-max-length` / `x-max-length-bytes`）

配置方式（业务队列声明时设置 DLX）：

```java
args.put("x-dead-letter-exchange", "dlx.exchange");
args.put("x-dead-letter-routing-key", "dlx.order");
```

并准备 DLX 侧的队列：

```java
channel.exchangeDeclare("dlx.exchange", "direct", true);
channel.queueDeclare("dlx.queue", true, false, false, null);
channel.queueBind("dlx.queue", "dlx.exchange", "dlx.order");
```

## 队列长度限制与溢出策略

- `x-max-length`：最大消息数
- `x-max-length-bytes`：最大字节数
- `x-overflow`：溢出策略
  - `reject-publish`：拒绝发布（推荐更安全）
  - `drop-head`：丢弃队头消息

```java
args.put("x-max-length", 100000);
args.put("x-overflow", "reject-publish");
```

## Lazy Queue（延迟加载队列）

`x-queue-mode=lazy` 将尽量把消息放磁盘，适合“消息堆积大但允许较高延迟”的场景。

```java
args.put("x-queue-mode", "lazy");
```

## 优先级队列

```java
args.put("x-max-priority", 10);
```

只有在积压时优先级更明显，且会增加资源开销。

## 队列类型：Classic vs Quorum vs Stream

## Classic（经典队列）

- 默认类型
- 性能好，功能成熟
- 适合多数场景

## Quorum（仲裁队列，推荐新项目优先）

- 基于 Raft
- 更强一致性与更稳定的故障恢复

```java
args.put("x-queue-type", "quorum");
```

## Stream（流队列）

- 适合大吞吐、保留较长历史、以及“可回溯消费”的场景

```java
args.put("x-queue-type", "stream");
args.put("x-max-length-bytes", 20_000_000_000L);
```

## 与消费确认（ack）联动的要点

- 使用手动 ack 时，`messages_unacknowledged` 会增长；预取值过大可能导致大量 unacked 占用内存
- 处理失败时：
  - `requeue=true` 可能导致“毒丸消息”反复重试
  - 更稳妥的是结合 DLX/重试队列/延迟重试（见 `message-types` 与 `consumer`）

## 下一步

- 💻 [交换机详解](/docs/rabbitmq/exchanges) - 先搞懂路由，再配置队列
- 🎯 [消息类型详解](/docs/rabbitmq/message-types) - TTL+DLX 延迟、优先级、幂等等
- 📊 [消费者指南](/docs/rabbitmq/consumer) - 手动 ack、nack/requeue 与重试策略
