---
sidebar_position: 6
title: "消费者指南"
description: "RabbitMQ 消费者开发指南"
---

# RabbitMQ 消费者指南

本指南详细介绍 RabbitMQ 消费者的开发和最佳实践。

## 消费者基础

### 推送模式（Push）

```java
import com.rabbitmq.client.*;

public class PushConsumer {
    public static void main(String[] args) throws Exception {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");

        Connection connection = factory.newConnection();
        Channel channel = connection.createChannel();

        // 声明队列
        channel.queueDeclare("my-queue", true, false, false, null);

        // 创建消费者回调
        DeliverCallback deliverCallback = (consumerTag, delivery) -> {
            String message = new String(delivery.getBody(), "UTF-8");
            System.out.println("Received: " + message);

            // 处理消息...

            // 手动确认
            channel.basicAck(delivery.getEnvelope().getDeliveryTag(), false);
        };

        // 取消回调
        CancelCallback cancelCallback = consumerTag -> {
            System.out.println("Consumer cancelled: " + consumerTag);
        };

        // 开始消费（手动确认模式）
        boolean autoAck = false;
        channel.basicConsume("my-queue", autoAck, deliverCallback, cancelCallback);
    }
}
```

### 拉取模式（Pull）

```java
// 单条拉取
GetResponse response = channel.basicGet("my-queue", false);
if (response != null) {
    String message = new String(response.getBody(), "UTF-8");
    System.out.println("Received: " + message);

    // 确认消息
    channel.basicAck(response.getEnvelope().getDeliveryTag(), false);
}
```

## 消息确认

### 自动确认

```java
// 自动确认（不推荐用于重要消息）
boolean autoAck = true;
channel.basicConsume("queue", autoAck, deliverCallback, cancelCallback);
```

### 手动确认

```java
// 单条确认
channel.basicAck(delivery.getEnvelope().getDeliveryTag(), false);

// 批量确认（确认该 tag 及之前所有未确认的消息）
channel.basicAck(delivery.getEnvelope().getDeliveryTag(), true);
```

### 拒绝消息

```java
// 拒绝单条消息并重新入队
channel.basicNack(deliveryTag, false, true);

// 拒绝单条消息不重新入队
channel.basicReject(deliveryTag, false);

// 批量拒绝
channel.basicNack(deliveryTag, true, true);
```

## QoS 预取

控制消费者一次能接收多少条未确认消息：

```java
// 设置预取数量（每个消费者）
int prefetchCount = 10;
channel.basicQos(prefetchCount);

// 全局设置（所有消费者共享）
channel.basicQos(prefetchCount, true);

// 设置预取大小和数量
int prefetchSize = 0;  // 0 表示不限制大小
channel.basicQos(prefetchSize, prefetchCount, false);
```

## 消费者完整示例

```java
public class RobustConsumer {
    private final Connection connection;
    private final Channel channel;
    private final String queueName;

    public RobustConsumer(String queueName) throws Exception {
        this.queueName = queueName;

        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");
        factory.setAutomaticRecoveryEnabled(true);
        factory.setNetworkRecoveryInterval(5000);

        this.connection = factory.newConnection();
        this.channel = connection.createChannel();

        // 设置 QoS
        channel.basicQos(10);

        // 声明队列
        channel.queueDeclare(queueName, true, false, false, null);
    }

    public void startConsuming() throws Exception {
        DeliverCallback deliverCallback = (consumerTag, delivery) -> {
            long deliveryTag = delivery.getEnvelope().getDeliveryTag();

            try {
                String message = new String(delivery.getBody(), "UTF-8");
                processMessage(message);

                // 处理成功，确认
                channel.basicAck(deliveryTag, false);

            } catch (Exception e) {
                // 处理失败，判断是否需要重试
                if (delivery.getEnvelope().isRedeliver()) {
                    // 已经是重发的消息，不再重试
                    channel.basicNack(deliveryTag, false, false);
                    // 可以发送到死信队列
                } else {
                    // 第一次失败，重新入队
                    channel.basicNack(deliveryTag, false, true);
                }
            }
        };

        CancelCallback cancelCallback = consumerTag -> {
            System.out.println("Consumer was cancelled");
        };

        String consumerTag = channel.basicConsume(queueName, false, deliverCallback, cancelCallback);
        System.out.println("Consumer started: " + consumerTag);
    }

    private void processMessage(String message) {
        // 业务处理逻辑
        System.out.println("Processing: " + message);
    }

    public void close() throws Exception {
        channel.close();
        connection.close();
    }
}
```

## 死信队列

### 配置死信队列

```java
// 声明死信交换机和队列
channel.exchangeDeclare("dlx-exchange", "direct", true);
channel.queueDeclare("dlx-queue", true, false, false, null);
channel.queueBind("dlx-queue", "dlx-exchange", "dlx-routing-key");

// 配置业务队列的死信设置
Map<String, Object> args = new HashMap<>();
args.put("x-dead-letter-exchange", "dlx-exchange");
args.put("x-dead-letter-routing-key", "dlx-routing-key");
args.put("x-message-ttl", 60000);  // 可选：消息 TTL

channel.queueDeclare("business-queue", true, false, false, args);
```

### 消息进入死信的情况

1. 消息被拒绝（basicReject/basicNack）且 requeue=false
2. 消息 TTL 过期
3. 队列达到最大长度

## Spring Boot 消费者

### 基本消费者

```java
@Component
@Slf4j
public class OrderConsumer {

    @RabbitListener(queues = "order-queue")
    public void handleOrder(Order order) {
        log.info("收到订单: {}", order.getId());
        // 处理订单逻辑
    }
}
```

### 手动确认

```java
@Component
@Slf4j
public class ManualAckConsumer {

    @RabbitListener(queues = "order-queue", ackMode = "MANUAL")
    public void handleOrder(Order order, Channel channel,
                           @Header(AmqpHeaders.DELIVERY_TAG) long deliveryTag) {
        try {
            // 处理业务逻辑
            processOrder(order);

            // 手动确认
            channel.basicAck(deliveryTag, false);
            log.info("订单处理成功: {}", order.getId());

        } catch (Exception e) {
            try {
                // 重新入队
                channel.basicNack(deliveryTag, false, true);
                log.error("订单处理失败，重新入队: {}", order.getId(), e);
            } catch (IOException ex) {
                log.error("Nack失败", ex);
            }
        }
    }
}
```

### 批量消费

```java
@RabbitListener(queues = "batch-queue", containerFactory = "batchContainerFactory")
public void handleBatch(List<Order> orders) {
    log.info("收到批量订单, 数量: {}", orders.size());
    for (Order order : orders) {
        processOrder(order);
    }
}

// 配置批量容器工厂
@Bean
public SimpleRabbitListenerContainerFactory batchContainerFactory(
        ConnectionFactory connectionFactory) {
    SimpleRabbitListenerContainerFactory factory = new SimpleRabbitListenerContainerFactory();
    factory.setConnectionFactory(connectionFactory);
    factory.setBatchListener(true);
    factory.setBatchSize(10);
    factory.setConsumerBatchEnabled(true);
    return factory;
}
```

### 并发消费者

```java
@RabbitListener(queues = "concurrent-queue", concurrency = "5-10")
public void handleConcurrent(String message) {
    log.info("Thread: {}, Message: {}", Thread.currentThread().getName(), message);
}

// 或者使用配置
@Bean
public SimpleRabbitListenerContainerFactory rabbitListenerContainerFactory(
        ConnectionFactory connectionFactory) {
    SimpleRabbitListenerContainerFactory factory = new SimpleRabbitListenerContainerFactory();
    factory.setConnectionFactory(connectionFactory);
    factory.setConcurrentConsumers(5);
    factory.setMaxConcurrentConsumers(10);
    factory.setPrefetchCount(10);
    return factory;
}
```

## 消息重试

### Spring Retry 配置

```yaml
spring:
  rabbitmq:
    listener:
      simple:
        retry:
          enabled: true
          initial-interval: 1000
          max-attempts: 3
          max-interval: 10000
          multiplier: 2
```

### 自定义重试

```java
@Component
public class RetryConsumer {

    private static final int MAX_RETRIES = 3;

    @RabbitListener(queues = "retry-queue", ackMode = "MANUAL")
    public void handle(Message message, Channel channel,
                      @Header(AmqpHeaders.DELIVERY_TAG) long deliveryTag) throws IOException {

        Integer retryCount = message.getMessageProperties().getHeader("x-retry-count");
        if (retryCount == null) retryCount = 0;

        try {
            processMessage(message);
            channel.basicAck(deliveryTag, false);

        } catch (Exception e) {
            if (retryCount < MAX_RETRIES) {
                // 重新发送带重试计数的消息
                retryCount++;
                message.getMessageProperties().setHeader("x-retry-count", retryCount);
                // 发送到延迟队列后重新消费
                channel.basicAck(deliveryTag, false);
                // 发送延迟消息...
            } else {
                // 达到最大重试次数，进入死信
                channel.basicNack(deliveryTag, false, false);
            }
        }
    }
}
```

## 消费者最佳实践

### 1. 幂等性处理

```java
@Service
public class IdempotentConsumer {

    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    @RabbitListener(queues = "order-queue")
    public void handle(Order order, @Header("messageId") String messageId) {
        // 检查是否已处理
        String key = "processed:" + messageId;
        Boolean isNew = redisTemplate.opsForValue().setIfAbsent(key, "1", 24, TimeUnit.HOURS);

        if (Boolean.FALSE.equals(isNew)) {
            log.warn("消息已处理过: {}", messageId);
            return;
        }

        try {
            processOrder(order);
        } catch (Exception e) {
            // 处理失败，删除标记以便重试
            redisTemplate.delete(key);
            throw e;
        }
    }
}
```

### 2. 异常处理

```java
@Configuration
public class RabbitErrorConfig {

    @Bean
    public RabbitListenerErrorHandler customErrorHandler() {
        return (message, channel, exception) -> {
            log.error("消息处理异常", exception);
            // 发送告警
            // 记录失败消息
            return null;
        };
    }
}

@RabbitListener(queues = "queue", errorHandler = "customErrorHandler")
public void handle(String message) {
    // 处理逻辑
}
```

### 3. 优雅关闭

```java
@Component
public class GracefulShutdown {

    @Autowired
    private RabbitListenerEndpointRegistry registry;

    @PreDestroy
    public void shutdown() {
        // 停止接收新消息
        registry.stop();

        // 等待当前消息处理完成
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

## 性能优化

### 1. 合理设置预取值

```java
// 长时间处理的任务，降低预取值
channel.basicQos(1);

// 快速处理的任务，提高预取值
channel.basicQos(100);
```

### 2. 批量确认

```java
private int unackedCount = 0;
private final int batchSize = 10;

DeliverCallback callback = (consumerTag, delivery) -> {
    processMessage(delivery.getBody());
    unackedCount++;

    if (unackedCount >= batchSize) {
        channel.basicAck(delivery.getEnvelope().getDeliveryTag(), true);
        unackedCount = 0;
    }
};
```

### 3. 并发消费

```java
// 创建多个消费者
ExecutorService executor = Executors.newFixedThreadPool(5);
for (int i = 0; i < 5; i++) {
    executor.submit(() -> {
        Channel channel = connection.createChannel();
        channel.basicQos(10);
        channel.basicConsume("queue", false, deliverCallback, cancelCallback);
    });
}
```

## 下一步

- ⚙️ [集群管理](/docs/rabbitmq/cluster-management) - 学习集群部署和管理
- 🚀 [性能优化](/docs/rabbitmq/performance-optimization) - 优化消费性能
- ✨ [最佳实践](/docs/rabbitmq/best-practices) - 生产环境建议

## 参考资料

- [RabbitMQ 消费者指南](https://www.rabbitmq.com/consumers.html)
- [消费者确认](https://www.rabbitmq.com/confirms.html#consumer-acknowledgements)
