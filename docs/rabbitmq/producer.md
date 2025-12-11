---
sidebar_position: 5
title: "生产者指南"
description: "RabbitMQ 生产者开发指南"
---

# RabbitMQ 生产者指南

本指南详细介绍 RabbitMQ 生产者的开发和最佳实践。

## 生产者基础

### 连接和通道

```java
import com.rabbitmq.client.*;

public class ProducerExample {
    public static void main(String[] args) throws Exception {
        // 创建连接工厂
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");
        factory.setPort(5672);
        factory.setUsername("guest");
        factory.setPassword("guest");
        factory.setVirtualHost("/");

        // 连接配置
        factory.setConnectionTimeout(30000);      // 连接超时
        factory.setRequestedHeartbeat(60);        // 心跳间隔
        factory.setAutomaticRecoveryEnabled(true); // 自动恢复

        // 创建连接
        Connection connection = factory.newConnection("my-producer");

        // 创建通道
        Channel channel = connection.createChannel();
    }
}
```

## 发送消息

### 基本发送

```java
// 声明队列
channel.queueDeclare("my-queue", true, false, false, null);

// 发送简单消息
String message = "Hello World!";
channel.basicPublish("", "my-queue", null, message.getBytes("UTF-8"));
```

### 发送到交换机

```java
// 声明交换机
channel.exchangeDeclare("my-exchange", "direct", true);

// 声明队列并绑定
channel.queueDeclare("my-queue", true, false, false, null);
channel.queueBind("my-queue", "my-exchange", "routing-key");

// 发送消息
channel.basicPublish("my-exchange", "routing-key", null, message.getBytes());
```

### 消息属性

```java
// 构建消息属性
AMQP.BasicProperties properties = new AMQP.BasicProperties.Builder()
    .contentType("application/json")
    .contentEncoding("UTF-8")
    .deliveryMode(2)                    // 持久化
    .priority(5)                        // 优先级 0-9
    .correlationId(UUID.randomUUID().toString())
    .replyTo("reply-queue")
    .expiration("60000")                // TTL 60秒
    .messageId(UUID.randomUUID().toString())
    .timestamp(new Date())
    .type("order.created")
    .userId("guest")
    .appId("order-service")
    .headers(Map.of("custom-header", "value"))
    .build();

channel.basicPublish("my-exchange", "routing-key", properties, message.getBytes());
```

## 消息持久化

### 配置持久化

```java
// 1. 声明持久化队列
boolean durable = true;
channel.queueDeclare("durable-queue", durable, false, false, null);

// 2. 声明持久化交换机
channel.exchangeDeclare("durable-exchange", "direct", true);

// 3. 发送持久化消息
AMQP.BasicProperties props = MessageProperties.PERSISTENT_TEXT_PLAIN;
channel.basicPublish("", "durable-queue", props, message.getBytes());
```

## 发布确认

### 单条确认

```java
// 启用发布确认
channel.confirmSelect();

// 发送消息
channel.basicPublish("", "queue", null, message.getBytes());

// 等待确认
if (channel.waitForConfirms(5000)) {
    System.out.println("消息已确认");
} else {
    System.out.println("消息未确认");
}
```

### 批量确认

```java
channel.confirmSelect();

int batchSize = 100;
int outstandingMessageCount = 0;

for (int i = 0; i < 1000; i++) {
    channel.basicPublish("", "queue", null, ("Message " + i).getBytes());
    outstandingMessageCount++;

    if (outstandingMessageCount >= batchSize) {
        channel.waitForConfirmsOrDie(5000);
        outstandingMessageCount = 0;
    }
}

// 确认剩余消息
if (outstandingMessageCount > 0) {
    channel.waitForConfirmsOrDie(5000);
}
```

### 异步确认

```java
channel.confirmSelect();

ConcurrentNavigableMap<Long, String> outstandingConfirms = new ConcurrentSkipListMap<>();

// 确认回调
ConfirmCallback ackCallback = (sequenceNumber, multiple) -> {
    if (multiple) {
        ConcurrentNavigableMap<Long, String> confirmed =
            outstandingConfirms.headMap(sequenceNumber, true);
        confirmed.clear();
    } else {
        outstandingConfirms.remove(sequenceNumber);
    }
    System.out.println("Message confirmed: " + sequenceNumber);
};

// 否认回调
ConfirmCallback nackCallback = (sequenceNumber, multiple) -> {
    String message = outstandingConfirms.get(sequenceNumber);
    System.err.println("Message nacked: " + sequenceNumber + ", msg: " + message);
    // 重发逻辑
};

channel.addConfirmListener(ackCallback, nackCallback);

// 发送消息
for (int i = 0; i < 1000; i++) {
    String message = "Message " + i;
    outstandingConfirms.put(channel.getNextPublishSeqNo(), message);
    channel.basicPublish("", "queue", null, message.getBytes());
}
```

## 消息返回

当消息无法路由时,可以获取返回通知:

```java
// 添加返回监听器
channel.addReturnListener((replyCode, replyText, exchange, routingKey, properties, body) -> {
    System.err.printf("消息返回: code=%d, text=%s, exchange=%s, routingKey=%s%n",
        replyCode, replyText, exchange, routingKey);
    // 处理无法路由的消息
});

// 发送消息时设置 mandatory 标志
boolean mandatory = true;
channel.basicPublish("my-exchange", "invalid-key", mandatory, null, message.getBytes());
```

## 交换机类型

### Direct 交换机

```java
// 声明 Direct 交换机
channel.exchangeDeclare("direct-exchange", BuiltinExchangeType.DIRECT, true);

// 绑定多个路由键
channel.queueBind("error-queue", "direct-exchange", "error");
channel.queueBind("warning-queue", "direct-exchange", "warning");
channel.queueBind("info-queue", "direct-exchange", "info");

// 发送到指定路由
channel.basicPublish("direct-exchange", "error", null, "Error message".getBytes());
```

### Fanout 交换机

```java
// 声明 Fanout 交换机
channel.exchangeDeclare("fanout-exchange", BuiltinExchangeType.FANOUT, true);

// 绑定队列（路由键被忽略）
channel.queueBind("queue1", "fanout-exchange", "");
channel.queueBind("queue2", "fanout-exchange", "");

// 广播消息
channel.basicPublish("fanout-exchange", "", null, "Broadcast message".getBytes());
```

### Topic 交换机

```java
// 声明 Topic 交换机
channel.exchangeDeclare("topic-exchange", BuiltinExchangeType.TOPIC, true);

// 使用通配符绑定
channel.queueBind("all-logs", "topic-exchange", "#");           // 所有消息
channel.queueBind("kern-logs", "topic-exchange", "kern.*");     // kern 开头
channel.queueBind("critical", "topic-exchange", "*.critical");  // critical 结尾

// 发送消息
channel.basicPublish("topic-exchange", "kern.critical", null, message.getBytes());
channel.basicPublish("topic-exchange", "app.info", null, message.getBytes());
```

## Spring Boot 生产者

### 配置类

```java
@Configuration
public class RabbitProducerConfig {

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory) {
        RabbitTemplate template = new RabbitTemplate(connectionFactory);
        template.setMessageConverter(jackson2JsonMessageConverter());

        // 发布确认回调
        template.setConfirmCallback((correlationData, ack, cause) -> {
            if (ack) {
                System.out.println("消息确认成功");
            } else {
                System.err.println("消息确认失败: " + cause);
            }
        });

        // 消息返回回调
        template.setReturnsCallback(returned -> {
            System.err.printf("消息返回: exchange=%s, routingKey=%s, replyCode=%d%n",
                returned.getExchange(), returned.getRoutingKey(), returned.getReplyCode());
        });

        return template;
    }

    @Bean
    public MessageConverter jackson2JsonMessageConverter() {
        return new Jackson2JsonMessageConverter();
    }
}
```

### 生产者服务

```java
@Service
@Slf4j
public class OrderMessageProducer {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    public void sendOrder(Order order) {
        CorrelationData correlationData = new CorrelationData(order.getId());

        rabbitTemplate.convertAndSend(
            "order-exchange",
            "order.created",
            order,
            message -> {
                message.getMessageProperties().setDeliveryMode(MessageDeliveryMode.PERSISTENT);
                message.getMessageProperties().setPriority(5);
                return message;
            },
            correlationData
        );

        log.info("订单消息已发送: {}", order.getId());
    }

    public void sendDelayedMessage(String message, long delayMs) {
        rabbitTemplate.convertAndSend(
            "delayed-exchange",
            "delayed-key",
            message,
            msg -> {
                msg.getMessageProperties().setDelay((int) delayMs);
                return msg;
            }
        );
    }
}
```

## 最佳实践

### 1. 连接管理

```java
// 使用连接池
public class ConnectionPool {
    private final ConnectionFactory factory;
    private final List<Connection> connections;
    private final int poolSize;

    public ConnectionPool(ConnectionFactory factory, int poolSize) {
        this.factory = factory;
        this.poolSize = poolSize;
        this.connections = new ArrayList<>(poolSize);
        initPool();
    }

    private void initPool() {
        for (int i = 0; i < poolSize; i++) {
            connections.add(factory.newConnection());
        }
    }

    public Connection getConnection() {
        // 轮询返回连接
        return connections.get(ThreadLocalRandom.current().nextInt(poolSize));
    }
}
```

### 2. 消息序列化

```java
// 使用 JSON 序列化
ObjectMapper mapper = new ObjectMapper();

public void sendJson(Object data) throws Exception {
    String json = mapper.writeValueAsString(data);

    AMQP.BasicProperties props = new AMQP.BasicProperties.Builder()
        .contentType("application/json")
        .build();

    channel.basicPublish("exchange", "key", props, json.getBytes("UTF-8"));
}
```

### 3. 错误处理

```java
public void sendWithRetry(String message, int maxRetries) {
    int retries = 0;
    while (retries < maxRetries) {
        try {
            channel.confirmSelect();
            channel.basicPublish("", "queue", null, message.getBytes());
            if (channel.waitForConfirms(5000)) {
                return; // 发送成功
            }
        } catch (Exception e) {
            retries++;
            if (retries >= maxRetries) {
                throw new RuntimeException("发送失败，已重试 " + maxRetries + " 次", e);
            }
            try {
                Thread.sleep(1000 * retries); // 指数退避
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
```

## 性能优化

### 批量发送

```java
// 批量发送提升性能
public void batchSend(List<String> messages) throws Exception {
    channel.confirmSelect();

    for (String msg : messages) {
        channel.basicPublish("", "queue", null, msg.getBytes());
    }

    channel.waitForConfirmsOrDie(10000);
}
```

### 通道复用

```java
// 复用通道而不是每次创建新通道
private final ThreadLocal<Channel> channelHolder = ThreadLocal.withInitial(() -> {
    try {
        return connection.createChannel();
    } catch (IOException e) {
        throw new RuntimeException(e);
    }
});

public void send(String message) throws Exception {
    Channel channel = channelHolder.get();
    channel.basicPublish("", "queue", null, message.getBytes());
}
```

## 下一步

- 📖 [消费者指南](./consumer.md) - 学习消费者开发
- ⚙️ [集群管理](./cluster-management.md) - 了解集群部署
- 🚀 [性能优化](./performance-optimization.md) - 优化生产者性能

## 参考资料

- [RabbitMQ 发布者指南](https://www.rabbitmq.com/publishers.html)
- [发布确认](https://www.rabbitmq.com/confirms.html)
