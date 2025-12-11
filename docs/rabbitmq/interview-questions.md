---
sidebar_position: 12
title: "面试题集"
description: "RabbitMQ 面试常见问题"
---

# RabbitMQ 面试题集

## 基础概念

### 1. 什么是 RabbitMQ？

RabbitMQ 是一个开源的消息代理软件,实现了 AMQP 协议,用于在分布式系统中进行消息传递。

**核心特点:**

- 可靠性:支持持久化、确认机制
- 灵活路由:多种交换机类型
- 高可用:集群和镜像队列
- 多协议:AMQP、STOMP、MQTT

### 2. RabbitMQ 的核心组件?

| 组件     | 说明            |
| -------- | --------------- |
| Producer | 消息生产者      |
| Exchange | 交换机,路由消息 |
| Binding  | 绑定规则        |
| Queue    | 消息队列        |
| Consumer | 消息消费者      |

### 3. 四种交换机类型的区别?

- **Direct**: 精确匹配路由键
- **Fanout**: 广播到所有绑定队列
- **Topic**: 模式匹配(`*` 匹配一个词,`#` 匹配多个词)
- **Headers**: 根据消息头属性匹配

### 4. 如何保证消息可靠性?

```java
// 1. 持久化队列
channel.queueDeclare("queue", true, false, false, null);

// 2. 持久化消息
channel.basicPublish("", "queue", MessageProperties.PERSISTENT_TEXT_PLAIN, msg);

// 3. 发布确认
channel.confirmSelect();
channel.waitForConfirms();

// 4. 手动确认
channel.basicAck(deliveryTag, false);
```

### 5. 消息确认机制?

**生产者确认:**

- `confirmSelect()`: 启用确认模式
- `waitForConfirms()`: 同步确认
- `addConfirmListener()`: 异步确认

**消费者确认:**

- `basicAck()`: 确认消息
- `basicNack()`: 拒绝消息
- `basicReject()`: 拒绝单条

## 进阶问题

### 6. 如何处理消息重复消费?

**幂等性方案:**

```java
// 使用 Redis 去重
String msgId = properties.getMessageId();
if (!redis.setIfAbsent("msg:" + msgId, "1", 24, TimeUnit.HOURS)) {
    return; // 已处理
}
processMessage(body);
```

### 7. 死信队列的使用场景?

消息进入死信的情况:

1. 被拒绝且不重新入队
2. 消息 TTL 过期
3. 队列达到最大长度

```java
Map<String, Object> args = new HashMap<>();
args.put("x-dead-letter-exchange", "dlx");
args.put("x-dead-letter-routing-key", "dead");
channel.queueDeclare("queue", true, false, false, args);
```

### 8. 如何实现延迟消息?

**方案 1: 延迟插件**

```java
headers.put("x-delay", 60000);
```

**方案 2: TTL + 死信**

```java
args.put("x-message-ttl", 60000);
args.put("x-dead-letter-exchange", "target");
```

### 9. 镜像队列和 Quorum 队列的区别?

| 特性     | 镜像队列 | Quorum 队列  |
| -------- | -------- | ------------ |
| 一致性   | 弱一致   | 强一致(Raft) |
| 性能     | 较高     | 中等         |
| 数据安全 | 一般     | 更好         |
| 推荐场景 | 老版本   | 3.8+新项目   |

### 10. 如何保证消息顺序?

1. 单队列单消费者
2. 业务分区:按业务键路由到固定队列
3. 消息携带序号,消费端排序

## 场景设计

### 11. 如何设计一个订单系统的消息队列?

```
订单服务 -> [订单交换机] -> 库存队列 -> 库存服务
                        -> 支付队列 -> 支付服务
                        -> 通知队列 -> 通知服务
```

**关键点:**

- 订单消息持久化
- 死信队列处理失败订单
- 幂等性保证

### 12. 高并发场景如何优化?

1. 多消费者并发消费
2. 调大预取值
3. 批量确认
4. 消息压缩
5. 集群部署

### 13. RabbitMQ vs Kafka?

| 特性   | RabbitMQ | Kafka  |
| ------ | -------- | ------ |
| 协议   | AMQP     | 自定义 |
| 吞吐量 | 中等     | 极高   |
| 延迟   | 微秒     | 毫秒   |
| 路由   | 灵活     | 简单   |
| 回溯   | 不支持   | 支持   |
| 场景   | 任务队列 | 日志流 |

## 代码题

### 14. 实现一个简单的生产者

```java
public class Producer {
    public static void main(String[] args) throws Exception {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");

        try (Connection conn = factory.newConnection();
             Channel channel = conn.createChannel()) {

            channel.queueDeclare("hello", true, false, false, null);
            channel.confirmSelect();

            String message = "Hello World!";
            channel.basicPublish("", "hello",
                MessageProperties.PERSISTENT_TEXT_PLAIN,
                message.getBytes());

            channel.waitForConfirms(5000);
        }
    }
}
```

### 15. 实现一个可靠的消费者

```java
public class Consumer {
    public static void main(String[] args) throws Exception {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");
        factory.setAutomaticRecoveryEnabled(true);

        Connection conn = factory.newConnection();
        Channel channel = conn.createChannel();

        channel.basicQos(10);

        DeliverCallback callback = (tag, delivery) -> {
            try {
                process(delivery.getBody());
                channel.basicAck(delivery.getEnvelope().getDeliveryTag(), false);
            } catch (Exception e) {
                channel.basicNack(delivery.getEnvelope().getDeliveryTag(), false, true);
            }
        };

        channel.basicConsume("hello", false, callback, tag -> {});
    }
}
```

## 参考资料

- [RabbitMQ 官方文档](https://www.rabbitmq.com/documentation.html)
