---
id: rocketmq
title: RocketMQ 消息队列
sidebar_label: RocketMQ
sidebar_position: 6
---

# RocketMQ 消息队列

> [!TIP] > **高性能消息中间件**: RocketMQ 是阿里巴巴开源的分布式消息中间件，支持事务消息、顺序消息、延迟消息等多种特性。

## 1. RocketMQ 简介

**RocketMQ** 是一款低延迟、高可靠、可伸缩、易使用的消息中间件。

### 核心特性

- **高吞吐** - 单机支持千万级消息堆积
- **事务消息** - 支持分布式事务
- **顺序消息** - 保证消息顺序
- **延迟消息** - 支持定时投递
- **消息过滤** - 支持 Tag 和 SQL 过滤

## 2. 核心概念

- **Producer** - 消息生产者
- **Consumer** - 消息消费者
- **Topic** - 消息主题
- **Tag** - 消息标签
- **Message Queue** - 消息队列
- **Broker** - 消息代理服务器
- **NameServer** - 路由中心

## 3. 快速开始

### 添加依赖

```xml
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-stream-rocketmq</artifactId>
</dependency>
```

或使用 RocketMQ Spring Boot Starter：

```xml
<dependency>
    <groupId>org.apache.rocketmq</groupId>
    <artifactId>rocketmq-spring-boot-starter</artifactId>
</dependency>
```

### 配置

```yaml
rocketmq:
  name-server: localhost:9876
  producer:
    group: producer-group
    send-message-timeout: 3000
    retry-times-when-send-failed: 2
```

### 发送消息

```java
@Service
public class OrderService {

    @Autowired
    private RocketMQTemplate rocketMQTemplate;

    public void createOrder(Order order) {
        // 同步发送
        rocketMQTemplate.convertAndSend("order-topic", order);

        // 异步发送
        rocketMQTemplate.asyncSend("order-topic", order, new SendCallback() {
            @Override
            public void onSuccess(SendResult sendResult) {
                System.out.println("发送成功");
            }

            @Override
            public void onException(Throwable throwable) {
                System.out.println("发送失败");
            }
        });

        // 单向发送（不关心结果）
        rocketMQTemplate.sendOneWay("order-topic", order);
    }
}
```

### 消费消息

```java
@Service
@RocketMQMessageListener(
    topic = "order-topic",
    consumerGroup = "order-consumer-group"
)
public class OrderConsumer implements RocketMQListener<Order> {

    @Override
    public void onMessage(Order order) {
        System.out.println("收到订单: " + order);
        // 处理业务逻辑
    }
}
```

## 4. 事务消息

### 工作原理

```
1. 发送半消息（Half Message）
2. 执行本地事务
3. 提交或回滚半消息
4. 如果长时间未收到确认，回查本地事务状态
```

### 示例

```java
@Service
public class OrderService {

    @Autowired
    private RocketMQTemplate rocketMQTemplate;

    public void createOrder(Order order) {
        rocketMQTemplate.sendMessageInTransaction(
            "order-topic",
            MessageBuilder.withPayload(order).build(),
            order
        );
    }
}

@RocketMQTransactionListener
public class OrderTransactionListener implements RocketMQLocalTransactionListener {

    @Override
    public RocketMQLocalTransactionState executeLocalTransaction(Message msg, Object arg) {
        try {
            Order order = (Order) arg;
            // 执行本地事务
            orderService.save(order);
            return RocketMQLocalTransactionState.COMMIT;
        } catch (Exception e) {
            return RocketMQLocalTransactionState.ROLLBACK;
        }
    }

    @Override
    public RocketMQLocalTransactionState checkLocalTransaction(Message msg) {
        // 回查本地事务状态
        String orderId = msg.getHeaders().get("orderId");
        boolean exists = orderRepository.existsById(orderId);
        return exists ?
            RocketMQLocalTransactionState.COMMIT :
            RocketMQLocalTransactionState.ROLLBACK;
    }
}
```

## 5. 顺序消息

### 全局顺序

所有消息严格按照 FIFO 顺序消费（吞吐量低）：

```java
rocketMQTemplate.syncSendOrderly("order-topic", order, orderId);
```

### 分区顺序

同一 orderId 的消息保证顺序：

```java
// 发送
rocketMQTemplate.syncSendOrderly(
    "order-topic",
    order,
    orderId  // 相同 orderId 发送到同一队列
);

// 消费
@RocketMQMessageListener(
    topic = "order-topic",
    consumerGroup = "order-consumer-group",
    consumeMode = ConsumeMode.ORDERLY
)
public class OrderConsumer implements RocketMQListener<Order> {
    // ...
}
```

## 6. 延迟消息

RocketMQ 支持 18 个固定的延迟级别：

```
1s 5s 10s 30s 1m 2m 3m 4m 5m 6m 7m 8m 9m 10m 20m 30m 1h 2h
```

```java
// 发送延迟消息（延迟级别3 = 10秒）
rocketMQTemplate.syncSend(
    "order-topic",
    MessageBuilder.withPayload(order).build(),
    3000,  // 超时时间
    3      // 延迟级别
);
```

## 7. 消息过滤

### Tag 过滤

```java
// 发送带 Tag 的消息
rocketMQTemplate.convertAndSend("order-topic:TagA", order);

// 消费指定 Tag 的消息
@RocketMQMessageListener(
    topic = "order-topic",
    selectorExpression = "TagA || TagB",
    consumerGroup = "order-consumer-group"
)
public class OrderConsumer implements RocketMQListener<Order> {
    // ...
}
```

### SQL 过滤

```java
// 发送带属性的消息
Message<Order> msg = MessageBuilder
    .withPayload(order)
    .setHeader("region", "hangzhou")
    .setHeader("price", 100)
    .build();
rocketMQTemplate.send("order-topic", msg);

// SQL 过滤消费
@RocketMQMessageListener(
    topic = "order-topic",
    selectorType = SelectorType.SQL92,
    selectorExpression = "region = 'hangzhou' AND price > 50",
    consumerGroup = "order-consumer-group"
)
public class OrderConsumer implements RocketMQListener<Order> {
    // ...
}
```

## 8. 消费模式

### 集群消费（默认）

```
同一消费组的多个消费者共同消费消息
每条消息只会被消费组中的一个消费者消费
```

### 广播消费

```
同一消费组的每个消费者都会消费所有消息
```

```java
@RocketMQMessageListener(
    topic = "order-topic",
    consumerGroup = "order-consumer-group",
    messageModel = MessageModel.BROADCASTING
)
public class OrderConsumer implements RocketMQListener<Order> {
    // ...
}
```

## 9. 最佳实践

### 消息幂等

```java
@Service
@RocketMQMessageListener(topic = "order-topic", consumerGroup = "order-consumer-group")
public class OrderConsumer implements RocketMQListener<Order> {

    @Autowired
    private RedisTemplate redisTemplate;

    @Override
    public void onMessage(Order order) {
        String msgId = order.getMsgId();

        // 检查是否已处理
        if (redisTemplate.hasKey(msgId)) {
            return;
        }

        // 处理业务
        processOrder(order);

        // 记录已处理
        redisTemplate.opsForValue().set(msgId, "1", 24, TimeUnit.HOURS);
    }
}
```

### 消息重试

```java
@RocketMQMessageListener(
    topic = "order-topic",
    consumerGroup = "order-consumer-group",
    maxReconsumeTimes = 3  // 最大重试次数
)
public class OrderConsumer implements RocketMQListener<Order> {
    // ...
}
```

### 死信队列

重试次数超过最大值后，消息进入死信队列：

```
原 Topic: order-topic
死信 Topic: %DLQ%order-consumer-group
```

## 10. 总结

| 特性     | 说明            |
| -------- | --------------- |
| 事务消息 | 支持分布式事务  |
| 顺序消息 | 保证消息顺序    |
| 延迟消息 | 定时投递        |
| 消息过滤 | Tag 和 SQL 过滤 |
| 高性能   | 单机千万级堆积  |

---

**关键要点**：

- RocketMQ 功能强大，性能优秀
- 支持事务消息解决分布式事务
- 注意消息幂等性
- 合理使用重试和死信队列

**下一步**：学习 [Dubbo RPC 框架](/docs/springcloud-alibaba/dubbo)
