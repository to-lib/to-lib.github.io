---
sidebar_position: 12
title: "Spring 集成"
description: "RabbitMQ 与 Spring/Spring Boot 集成详解"
---

# RabbitMQ Spring 集成

## 概述

Spring AMQP 提供了对 RabbitMQ 的全面支持，简化了消息发送、接收和配置。本文详细介绍 Spring Boot 中 RabbitMQ 的集成使用。

## 快速开始

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-amqp</artifactId>
</dependency>
```

### 基础配置

```yaml
# application.yml
spring:
  rabbitmq:
    host: localhost
    port: 5672
    username: guest
    password: guest
    virtual-host: /
```

## RabbitTemplate

### 基本使用

```java
@Service
public class MessageService {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    // 发送到队列
    public void sendToQueue(String message) {
        rabbitTemplate.convertAndSend("myQueue", message);
    }

    // 发送到交换机
    public void sendToExchange(String message) {
        rabbitTemplate.convertAndSend("myExchange", "routingKey", message);
    }

    // 发送并接收响应（RPC）
    public String sendAndReceive(String message) {
        return (String) rabbitTemplate.convertSendAndReceive("rpcQueue", message);
    }
}
```

### 发送对象

```java
@Data
@AllArgsConstructor
public class Order {
    private String orderId;
    private String userId;
    private BigDecimal amount;
}

@Service
public class OrderService {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    public void sendOrder(Order order) {
        // 自动序列化为 JSON
        rabbitTemplate.convertAndSend("order.exchange", "order.created", order);
    }
}
```

### 消息属性设置

```java
@Service
public class AdvancedMessageService {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    public void sendWithProperties(String message) {
        rabbitTemplate.convertAndSend("exchange", "key", message, msg -> {
            MessageProperties props = msg.getMessageProperties();
            props.setMessageId(UUID.randomUUID().toString());
            props.setTimestamp(new Date());
            props.setExpiration("60000");  // 60秒过期
            props.setPriority(5);
            props.setHeader("custom-header", "value");
            return msg;
        });
    }

    // 发送延迟消息（需要延迟插件）
    public void sendDelayedMessage(String message, int delayMs) {
        rabbitTemplate.convertAndSend("delayed.exchange", "delayed", message, msg -> {
            msg.getMessageProperties().setDelay(delayMs);
            return msg;
        });
    }
}
```

## @RabbitListener

### 基本消费

```java
@Component
public class MessageConsumer {

    @RabbitListener(queues = "myQueue")
    public void receive(String message) {
        System.out.println("收到消息: " + message);
    }
}
```

### 接收对象

```java
@Component
public class OrderConsumer {

    @RabbitListener(queues = "orderQueue")
    public void receiveOrder(Order order) {
        System.out.println("收到订单: " + order.getOrderId());
    }
}
```

### 获取消息详情

```java
@Component
public class DetailedConsumer {

    @RabbitListener(queues = "myQueue")
    public void receive(Message message, Channel channel,
                        @Header(AmqpHeaders.DELIVERY_TAG) long tag) {
        try {
            String body = new String(message.getBody());
            String messageId = message.getMessageProperties().getMessageId();

            System.out.println("消息ID: " + messageId);
            System.out.println("消息内容: " + body);

            // 手动确认
            channel.basicAck(tag, false);

        } catch (Exception e) {
            try {
                // 拒绝并重新入队
                channel.basicNack(tag, false, true);
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }
}
```

### 自动声明队列

```java
@Component
public class AutoDeclareConsumer {

    @RabbitListener(bindings = @QueueBinding(
        value = @Queue(value = "auto.queue", durable = "true"),
        exchange = @Exchange(value = "auto.exchange", type = ExchangeTypes.DIRECT),
        key = "auto.key"
    ))
    public void receive(String message) {
        System.out.println("收到: " + message);
    }
}
```

### 并发消费

```java
@Component
public class ConcurrentConsumer {

    // 固定并发数
    @RabbitListener(queues = "concurrentQueue", concurrency = "5")
    public void receive(String message) {
        System.out.println(Thread.currentThread().getName() + " 收到: " + message);
    }

    // 动态并发数（最小3，最大10）
    @RabbitListener(queues = "dynamicQueue", concurrency = "3-10")
    public void receiveDynamic(String message) {
        System.out.println("动态消费: " + message);
    }
}
```

## 消息确认

### 配置确认模式

```yaml
spring:
  rabbitmq:
    listener:
      simple:
        acknowledge-mode: manual # manual, auto, none
        prefetch: 10 # 预取数量
```

### 手动确认

```java
@Component
public class ManualAckConsumer {

    @RabbitListener(queues = "ackQueue")
    public void receive(Message message, Channel channel,
                        @Header(AmqpHeaders.DELIVERY_TAG) long tag) {
        try {
            processMessage(message);

            // 确认成功
            channel.basicAck(tag, false);

        } catch (RecoverableException e) {
            // 可恢复，重新入队
            channel.basicNack(tag, false, true);

        } catch (Exception e) {
            // 不可恢复，发送到死信
            channel.basicNack(tag, false, false);
        }
    }
}
```

### 发布确认

```yaml
spring:
  rabbitmq:
    publisher-confirm-type: correlated # none, simple, correlated
    publisher-returns: true
```

```java
@Configuration
public class RabbitConfirmConfig {

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory) {
        RabbitTemplate template = new RabbitTemplate(connectionFactory);

        // 发布确认回调
        template.setConfirmCallback((correlationData, ack, cause) -> {
            if (ack) {
                System.out.println("消息发送成功: " + correlationData.getId());
            } else {
                System.out.println("消息发送失败: " + cause);
                // 重试或记录
            }
        });

        // 消息返回回调（无法路由时）
        template.setReturnsCallback(returned -> {
            System.out.println("消息被退回: " + returned.getMessage());
            System.out.println("退回原因: " + returned.getReplyText());
        });

        template.setMandatory(true);

        return template;
    }
}
```

## 配置类

### 队列和交换机声明

```java
@Configuration
public class RabbitConfig {

    // 声明队列
    @Bean
    public Queue orderQueue() {
        return QueueBuilder.durable("order.queue")
            .withArgument("x-message-ttl", 60000)
            .withArgument("x-dead-letter-exchange", "dlx.exchange")
            .withArgument("x-dead-letter-routing-key", "dlx.order")
            .build();
    }

    // 声明交换机
    @Bean
    public DirectExchange orderExchange() {
        return ExchangeBuilder.directExchange("order.exchange")
            .durable(true)
            .build();
    }

    // 绑定
    @Bean
    public Binding orderBinding(Queue orderQueue, DirectExchange orderExchange) {
        return BindingBuilder.bind(orderQueue)
            .to(orderExchange)
            .with("order.key");
    }

    // Topic 交换机
    @Bean
    public TopicExchange topicExchange() {
        return new TopicExchange("topic.exchange");
    }

    // Fanout 交换机
    @Bean
    public FanoutExchange fanoutExchange() {
        return new FanoutExchange("fanout.exchange");
    }
}
```

### 死信队列配置

```java
@Configuration
public class DeadLetterConfig {

    @Bean
    public Queue dlxQueue() {
        return new Queue("dlx.queue", true);
    }

    @Bean
    public DirectExchange dlxExchange() {
        return new DirectExchange("dlx.exchange");
    }

    @Bean
    public Binding dlxBinding() {
        return BindingBuilder.bind(dlxQueue())
            .to(dlxExchange())
            .with("dlx.#");
    }

    @Bean
    public Queue businessQueue() {
        return QueueBuilder.durable("business.queue")
            .withArgument("x-dead-letter-exchange", "dlx.exchange")
            .withArgument("x-dead-letter-routing-key", "dlx.business")
            .build();
    }
}
```

### JSON 序列化配置

```java
@Configuration
public class MessageConverterConfig {

    @Bean
    public MessageConverter jsonMessageConverter() {
        return new Jackson2JsonMessageConverter();
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory,
                                         MessageConverter jsonMessageConverter) {
        RabbitTemplate template = new RabbitTemplate(connectionFactory);
        template.setMessageConverter(jsonMessageConverter);
        return template;
    }
}
```

## 重试机制

### 消费者重试

```yaml
spring:
  rabbitmq:
    listener:
      simple:
        retry:
          enabled: true
          initial-interval: 1000ms
          max-attempts: 3
          multiplier: 2.0
          max-interval: 10000ms
```

### 自定义重试策略

```java
@Configuration
public class RetryConfig {

    @Bean
    public SimpleRabbitListenerContainerFactory rabbitListenerContainerFactory(
            ConnectionFactory connectionFactory) {

        SimpleRabbitListenerContainerFactory factory =
            new SimpleRabbitListenerContainerFactory();
        factory.setConnectionFactory(connectionFactory);

        // 配置重试模板
        RetryTemplate retryTemplate = new RetryTemplate();
        ExponentialBackOffPolicy backOffPolicy = new ExponentialBackOffPolicy();
        backOffPolicy.setInitialInterval(1000);
        backOffPolicy.setMultiplier(2.0);
        backOffPolicy.setMaxInterval(10000);
        retryTemplate.setBackOffPolicy(backOffPolicy);

        // 重试次数
        SimpleRetryPolicy retryPolicy = new SimpleRetryPolicy();
        retryPolicy.setMaxAttempts(3);
        retryTemplate.setRetryPolicy(retryPolicy);

        factory.setRetryTemplate(retryTemplate);

        // 重试耗尽后发送到死信
        factory.setDefaultRequeueRejected(false);

        return factory;
    }
}
```

### 错误处理

```java
@Configuration
public class ErrorHandlerConfig {

    @Bean
    public SimpleRabbitListenerContainerFactory rabbitListenerContainerFactory(
            ConnectionFactory connectionFactory) {

        SimpleRabbitListenerContainerFactory factory =
            new SimpleRabbitListenerContainerFactory();
        factory.setConnectionFactory(connectionFactory);

        // 自定义错误处理器
        factory.setErrorHandler(throwable -> {
            log.error("消息处理失败", throwable);
            // 发送告警、记录日志等
        });

        return factory;
    }
}
```

## Spring Cloud Stream

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-stream-rabbit</artifactId>
</dependency>
```

### 函数式编程模型

```yaml
spring:
  cloud:
    stream:
      bindings:
        orderInput-in-0:
          destination: order-topic
          group: order-group
        orderOutput-out-0:
          destination: order-result
      rabbit:
        bindings:
          orderInput-in-0:
            consumer:
              acknowledge-mode: MANUAL
```

```java
@Configuration
public class StreamConfig {

    @Bean
    public Consumer<Order> orderInput() {
        return order -> {
            System.out.println("收到订单: " + order.getOrderId());
            processOrder(order);
        };
    }

    @Bean
    public Supplier<OrderResult> orderOutput() {
        return () -> new OrderResult("order-123", "SUCCESS");
    }

    @Bean
    public Function<Order, OrderResult> orderProcessor() {
        return order -> {
            // 处理订单并返回结果
            return new OrderResult(order.getOrderId(), "PROCESSED");
        };
    }
}
```

### 发送消息

```java
@Service
public class StreamMessageService {

    @Autowired
    private StreamBridge streamBridge;

    public void sendOrder(Order order) {
        streamBridge.send("orderOutput-out-0", order);
    }
}
```

## 高级配置

### 完整配置示例

```yaml
spring:
  rabbitmq:
    host: ${RABBITMQ_HOST:localhost}
    port: ${RABBITMQ_PORT:5672}
    username: ${RABBITMQ_USERNAME:guest}
    password: ${RABBITMQ_PASSWORD:guest}
    virtual-host: /

    # 连接配置
    connection-timeout: 30000
    requested-heartbeat: 60

    # 发布确认
    publisher-confirm-type: correlated
    publisher-returns: true

    # 消费者配置
    listener:
      simple:
        acknowledge-mode: manual
        prefetch: 10
        concurrency: 5
        max-concurrency: 20
        default-requeue-rejected: false
        retry:
          enabled: true
          initial-interval: 1000
          max-attempts: 3
          multiplier: 2
          max-interval: 10000

    # 缓存配置
    cache:
      channel:
        size: 25
        checkout-timeout: 1000
      connection:
        size: 1
        mode: CHANNEL

    # SSL 配置
    ssl:
      enabled: false
      # key-store: classpath:client.p12
      # key-store-password: password
```

### 多 RabbitMQ 实例

```java
@Configuration
public class MultiRabbitConfig {

    @Primary
    @Bean
    public ConnectionFactory primaryConnectionFactory() {
        CachingConnectionFactory factory = new CachingConnectionFactory();
        factory.setHost("primary-rabbit");
        factory.setPort(5672);
        factory.setUsername("guest");
        factory.setPassword("guest");
        return factory;
    }

    @Bean
    public ConnectionFactory secondaryConnectionFactory() {
        CachingConnectionFactory factory = new CachingConnectionFactory();
        factory.setHost("secondary-rabbit");
        factory.setPort(5672);
        factory.setUsername("guest");
        factory.setPassword("guest");
        return factory;
    }

    @Primary
    @Bean
    public RabbitTemplate primaryRabbitTemplate(
            @Qualifier("primaryConnectionFactory") ConnectionFactory cf) {
        return new RabbitTemplate(cf);
    }

    @Bean
    public RabbitTemplate secondaryRabbitTemplate(
            @Qualifier("secondaryConnectionFactory") ConnectionFactory cf) {
        return new RabbitTemplate(cf);
    }
}
```

## 最佳实践

### 消息设计

```java
@Data
public class MessageWrapper<T> {
    private String messageId;
    private String messageType;
    private LocalDateTime timestamp;
    private String version;
    private T payload;
    private Map<String, String> metadata;

    public static <T> MessageWrapper<T> of(String type, T payload) {
        MessageWrapper<T> wrapper = new MessageWrapper<>();
        wrapper.setMessageId(UUID.randomUUID().toString());
        wrapper.setMessageType(type);
        wrapper.setTimestamp(LocalDateTime.now());
        wrapper.setVersion("1.0");
        wrapper.setPayload(payload);
        wrapper.setMetadata(new HashMap<>());
        return wrapper;
    }
}
```

### 幂等性处理

```java
@Service
public class IdempotentConsumer {

    @Autowired
    private StringRedisTemplate redisTemplate;

    @RabbitListener(queues = "orderQueue")
    public void handleOrder(Message message, Channel channel,
                            @Header(AmqpHeaders.DELIVERY_TAG) long tag) throws IOException {
        String messageId = message.getMessageProperties().getMessageId();
        String key = "processed:" + messageId;

        try {
            // 幂等性检查
            Boolean isNew = redisTemplate.opsForValue()
                .setIfAbsent(key, "1", Duration.ofHours(24));

            if (Boolean.FALSE.equals(isNew)) {
                channel.basicAck(tag, false);
                return;
            }

            // 处理业务
            processOrder(message);

            channel.basicAck(tag, false);

        } catch (Exception e) {
            redisTemplate.delete(key);
            channel.basicNack(tag, false, true);
        }
    }
}
```

## 参考资料

- [Spring AMQP 官方文档](https://docs.spring.io/spring-amqp/docs/current/reference/html/)
- [Spring Cloud Stream RabbitMQ Binder](https://docs.spring.io/spring-cloud-stream-binder-rabbit/docs/current/reference/html/)
- [Spring Boot RabbitMQ 配置](https://docs.spring.io/spring-boot/docs/current/reference/html/application-properties.html#application-properties.integration.spring.rabbitmq)
