---
sidebar_position: 12
title: "高级特性"
description: "RabbitMQ 可靠性、延迟/重试、DLX 与常见架构模式"
---

# 高级特性

本页把 RabbitMQ 的一些“组合拳”能力串起来：确认与可靠性、延迟与重试、死信、以及典型的业务落地模式。

## 可靠性三件套

## 1) 消息持久化

- durable exchange
- durable queue
- persistent message（`deliveryMode=2`）

缺一不可。

## 2) 发布确认（Publisher Confirms）

解决“生产者发出去到底有没有被 Broker 接收/落盘”的不确定性。

- 同步 confirm：简单但吞吐低
- 异步 confirm：生产常用

（实现代码见 `producer`）

## 3) 消费确认（Consumer Acks）

- 自动 ack：吞吐高但可能丢消息
- 手动 ack：更可靠，但需要处理 nack/requeue 与幂等

（实现代码见 `consumer`）

## 延迟消息

RabbitMQ 原生不直接提供“延迟队列”语义，但可通过：

- **TTL + DLX**（通用方案）
- **延迟消息插件**（体验更好）

（详见 `message-types`）

## 重试（Retry）与“毒丸消息”治理

### 常见反模式：一直 requeue=true

如果消费代码对某类数据永远无法处理（毒丸消息），`requeue=true` 会导致它无限循环。

### 推荐模式：重试队列 + 延迟 + 死信

一个常见拓扑：

- `business.queue`（正常消费）
- `retry.10s.queue` / `retry.1m.queue`（延迟重试）
- `dlq.queue`（最终失败收敛）

实现方式：

- retry 队列设置 `x-message-ttl`，并设置 DLX 指向业务交换机
- 业务队列设置 DLX 指向最终 DLQ

```mermaid
graph LR
  P[Producer] --> EX[Exchange]
  EX --> BQ[business.queue]
  BQ -->|nack requeue=false| REX[retry.exchange]
  REX --> RQ[retry.queue TTL]
  RQ -->|expired -> DLX| EX
  BQ -->|reject/expired|maxlen| DLX[dlx.exchange]
  DLX --> DLQ[dlq.queue]
```

## 幂等性（Idempotency）

在“至少一次投递（at-least-once）”模型下，重复消费是常态，需要在业务层保证幂等：

- 数据库唯一约束（messageId/orderId）
- Redis SETNX
- 幂等表 + 事务

（示例可参考 `consumer` 与 `message-types`）

## 顺序性（Ordering）

RabbitMQ 只保证：**同一队列内、单通道投递、单消费者顺序消费**时顺序相对稳定。

要实现业务顺序：

- 按业务键路由到固定队列（例如用户维度、订单维度）
- 单队列单消费者（牺牲吞吐换顺序）
- 携带 sequence，在消费端排序/校验

## RPC 模式

RabbitMQ 支持 RPC（request-reply），但要注意超时、消费者扩容、以及 reply 队列的治理。

Spring 中通常使用 `RabbitTemplate#convertSendAndReceive` 或更明确的消息模型。

## 下一步

- ⚙️ [高级配置](/docs/rabbitmq/advanced-config) - policy/limits/definitions
- 🎯 [消息类型详解](/docs/rabbitmq/message-types) - TTL/DLX/优先级/延迟
- ✨ [最佳实践](/docs/rabbitmq/best-practices) - 生产落地清单
