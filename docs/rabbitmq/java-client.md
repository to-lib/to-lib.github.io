---
sidebar_position: 6
title: "Java 客户端"
description: "RabbitMQ Java 客户端（amqp-client）连接、发布、消费与确认"
---

# Java 客户端

本文聚焦 RabbitMQ Java 原生客户端 `com.rabbitmq:amqp-client`（非 Spring）。它更贴近协议，适合理解底层行为与排障。

## Maven 依赖

```xml
<dependency>
  <groupId>com.rabbitmq</groupId>
  <artifactId>amqp-client</artifactId>
  <version>5.20.0</version>
</dependency>
```

## 连接（Connection）与通道（Channel）

```java
ConnectionFactory factory = new ConnectionFactory();
factory.setHost("localhost");
factory.setPort(5672);
factory.setUsername("guest");
factory.setPassword("guest");
factory.setVirtualHost("/");

factory.setRequestedHeartbeat(60);
factory.setConnectionTimeout(30_000);
factory.setAutomaticRecoveryEnabled(true);

try (Connection connection = factory.newConnection("java-client");
     Channel channel = connection.createChannel()) {

    channel.queueDeclare("q.demo", true, false, false, null);
    channel.basicPublish("", "q.demo", null, "hello".getBytes());
}
```

## 生产者：发布确认（Publisher Confirms）

建议在可靠性要求高时开启 confirm：

```java
channel.confirmSelect();
channel.basicPublish("", "q.demo", null, "msg".getBytes());
if (!channel.waitForConfirms(5_000)) {
    throw new RuntimeException("publish not confirmed");
}
```

更高吞吐建议使用异步 confirm（见 `producer` 页面）。

## mandatory / ReturnListener（无法路由时回调）

当消息到达交换机但无法路由到任何队列时：

```java
channel.addReturnListener((replyCode, replyText, exchange, routingKey, properties, body) -> {
    System.err.println("Returned: " + replyText);
});

channel.basicPublish("some.exchange", "no.such.key", true, null, "msg".getBytes());
```

另一种是使用备用交换机（见 `exchanges`）。

## 消费者：手动确认（Manual Ack）

```java
channel.basicQos(10);

DeliverCallback callback = (tag, delivery) -> {
    long deliveryTag = delivery.getEnvelope().getDeliveryTag();
    try {
        // 业务处理
        channel.basicAck(deliveryTag, false);
    } catch (Exception e) {
        // 可选择 requeue=true 进行重试，但要避免毒丸消息无限重试
        channel.basicNack(deliveryTag, false, false);
    }
};

channel.basicConsume("q.demo", false, callback, tag -> {});
```

## basicReject vs basicNack

- `basicReject(tag, requeue)`：单条拒绝
- `basicNack(tag, multiple, requeue)`：可批量拒绝

一般用 `basicNack` 更灵活。

## Pull 模式：basicGet

不推荐高吞吐场景，但适合调试或低频任务：

```java
GetResponse resp = channel.basicGet("q.demo", false);
if (resp != null) {
    try {
        // process
        channel.basicAck(resp.getEnvelope().getDeliveryTag(), false);
    } catch (Exception e) {
        channel.basicNack(resp.getEnvelope().getDeliveryTag(), false, true);
    }
}
```

## 常见坑

- **Channel 非线程安全**：每个线程使用独立 Channel
- **连接数过多**：优先复用 Connection，用多 Channel
- **自动恢复不等于业务级重试**：连接恢复后仍需考虑“消息是否重复/是否丢失”的业务语义

## 下一步

- 💻 [生产者指南](/docs/rabbitmq/producer) - confirm/return/序列化/重试
- 📊 [消费者指南](/docs/rabbitmq/consumer) - ack/nack/requeue、预取与重试
- 🎯 [核心概念](/docs/rabbitmq/core-concepts) - Connection/Channel/Queue/Exchange
