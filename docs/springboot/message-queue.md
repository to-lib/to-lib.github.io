---
sidebar_position: 18
---

# 消息队列

> [!IMPORTANT]
> **异步通信核心**: 消息队列用于系统解耦和异步处理。本章介绍 RabbitMQ 和 Kafka 集成,注意消息可靠性和幂等性。

## RabbitMQ 集成

### 依赖配置

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-amqp</artifactId>
</dependency>
```

### RabbitMQ 配置

```yaml
spring:
  rabbitmq:
    host: localhost
    port: 5672
    username: guest
    password: guest
    virtual-host: /
    
    # 连接配置
    connection-timeout: 10000
    
    # 发布者配置
    publisher-confirms: true
    publisher-returns: true
    
    # 消费者配置
    listener:
      simple:
        prefetch: 1
        acknowledge-mode: manual
        concurrency: 5
        max-concurrency: 10
```

### 消息生产者

```java
import org.springframework.amqp.core.*;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Service;
import lombok.extern.slf4j.Slf4j;

// 配置队列和交换机
@Configuration
public class RabbitConfig {
    
    // 定义队列
    @Bean
    public Queue userQueue() {
        return new Queue("user.queue", true);  // 持久化队列
    }
    
    // 定义交换机
    @Bean
    public TopicExchange userExchange() {
        return new TopicExchange("user.exchange", true, false);
    }
    
    // 绑定队列和交换机
    @Bean
    public Binding userBinding(Queue userQueue, TopicExchange userExchange) {
        return BindingBuilder.bind(userQueue)
            .to(userExchange)
            .with("user.#");
    }
}

// 消息生产者
@Service
@RequiredArgsConstructor
@Slf4j
public class UserMessageProducer {
    
    private final RabbitTemplate rabbitTemplate;
    
    public void sendUserMessage(User user) {
        try {
            rabbitTemplate.convertAndSend("user.exchange", "user.created", user,
                message -> {
                    message.getMessageProperties().setDeliveryMode(MessageDeliveryMode.PERSISTENT);
                    return message;
                });
            log.info("User message sent successfully: {}", user.getId());
        } catch (Exception e) {
            log.error("Failed to send user message", e);
        }
    }
}
```

### 消息消费者

```java
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Service;
import com.rabbitmq.client.Channel;
import org.springframework.amqp.support.AmqpHeaders;
import org.springframework.messaging.handler.annotation.Header;
import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class UserMessageConsumer {
    
    // 监听队列消息
    @RabbitListener(queues = "user.queue")
    public void handleUserMessage(User user, 
                                 Channel channel,
                                 @Header(AmqpHeaders.DELIVERY_TAG) long deliveryTag) {
        try {
            log.info("Processing user message: {}", user.getId());
            
            // 处理业务逻辑
            processUser(user);
            
            // 手动确认
            channel.basicAck(deliveryTag, false);
        } catch (Exception e) {
            try {
                // 重新入队
                channel.basicNack(deliveryTag, false, true);
            } catch (IOException ex) {
                log.error("Failed to nack message", ex);
            }
        }
    }
    
    private void processUser(User user) {
        // 业务处理
    }
}
```

## Apache Kafka 集成

### 依赖配置

```xml
<dependency>
    <groupId>org.springframework.kafka</groupId>
    <artifactId>spring-kafka</artifactId>
</dependency>
```

### Kafka 配置

```yaml
spring:
  kafka:
    bootstrap-servers: localhost:9092
    
    # 生产者配置
    producer:
      key-serializer: org.apache.kafka.common.serialization.StringSerializer
      value-serializer: org.springframework.kafka.support.serializer.JsonSerializer
      acks: all
      retries: 3
      batch-size: 16384
      linger-ms: 10
    
    # 消费者配置
    consumer:
      bootstrap-servers: localhost:9092
      group-id: user-service-group
      key-deserializer: org.apache.kafka.common.serialization.StringDeserializer
      value-deserializer: org.springframework.kafka.support.serializer.JsonDeserializer
      auto-offset-reset: earliest
      enable-auto-commit: false
      max-poll-records: 100
    
    # 监听器配置
    listener:
      ack-mode: manual
      concurrency: 5
```

### Kafka 生产者

```java
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.KafkaHeaders;
import org.springframework.messaging.Message;
import org.springframework.messaging.support.MessageBuilder;
import org.springframework.stereotype.Service;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserEventProducer {
    
    private final KafkaTemplate<String, User> kafkaTemplate;
    
    public void sendUserEvent(User user, String eventType) {
        try {
            Message<User> message = MessageBuilder
                .withPayload(user)
                .setHeader(KafkaHeaders.TOPIC, "user-events")
                .setHeader(KafkaHeaders.MESSAGE_KEY, user.getId().toString())
                .setHeader("event-type", eventType)
                .setHeader("timestamp", System.currentTimeMillis())
                .build();
            
            kafkaTemplate.send(message).get();
            log.info("User event sent: {}", user.getId());
        } catch (Exception e) {
            log.error("Failed to send user event", e);
        }
    }
}
```

### Kafka 消费者

```java
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.rebalance.ConsumerSeekAware;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.kafka.support.KafkaHeaders;
import org.springframework.messaging.handler.annotation.Header;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Service;
import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class UserEventConsumer implements ConsumerSeekAware {
    
    @KafkaListener(
        topics = "user-events",
        groupId = "user-service-group",
        containerFactory = "kafkaListenerContainerFactory"
    )
    public void handleUserEvent(
            @Payload User user,
            @Header(KafkaHeaders.RECEIVED_PARTITION_ID) int partition,
            @Header(KafkaHeaders.OFFSET) long offset,
            @Header("event-type") String eventType,
            Acknowledgment acknowledgment) {
        
        try {
            log.info("Processing user event - ID: {}, Type: {}, Partition: {}, Offset: {}",
                user.getId(), eventType, partition, offset);
            
            // 处理事件
            processUserEvent(user, eventType);
            
            // 手动提交偏移量
            acknowledgment.acknowledge();
        } catch (Exception e) {
            log.error("Failed to process user event", e);
        }
    }
    
    private void processUserEvent(User user, String eventType) {
        switch (eventType) {
            case "USER_CREATED":
                handleUserCreated(user);
                break;
            case "USER_UPDATED":
                handleUserUpdated(user);
                break;
            case "USER_DELETED":
                handleUserDeleted(user);
                break;
        }
    }
    
    private void handleUserCreated(User user) {
        // 处理用户创建事件
    }
    
    private void handleUserUpdated(User user) {
        // 处理用户更新事件
    }
    
    private void handleUserDeleted(User user) {
        // 处理用户删除事件
    }
}
```

## 事件驱动架构

### 基于 Spring Events 的事件处理

```java
import org.springframework.context.ApplicationEvent;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

// 定义事件
@Getter
public class UserCreatedEvent extends ApplicationEvent {
    private final User user;
    
    public UserCreatedEvent(Object source, User user) {
        super(source);
        this.user = user;
    }
}

// 事件发布者
@Service
@RequiredArgsConstructor
public class UserService {
    
    private final ApplicationEventPublisher eventPublisher;
    
    public User createUser(User user) {
        // 创建用户
        User savedUser = userRepository.save(user);
        
        // 发布事件
        eventPublisher.publishEvent(new UserCreatedEvent(this, savedUser));
        
        return savedUser;
    }
}

// 事件监听器
@Service
public class UserEventListeners {
    
    @EventListener
    public void onUserCreated(UserCreatedEvent event) {
        User user = event.getUser();
        log.info("User created: {}", user.getId());
        
        // 发送欢迎邮件
        sendWelcomeEmail(user);
    }
    
    private void sendWelcomeEmail(User user) {
        // 发送邮件逻辑
    }
}
```

## 消息队列选择

### RabbitMQ vs Kafka

| 特性 | RabbitMQ | Kafka |
|------|----------|-------|
| 吞吐量 | 中等 | 高 |
| 延迟 | 低 | 中等 |
| 持久化 | 支持 | 支持 |
| 消费者 | 单消费 | 消费组 |
| 使用场景 | 任务队列、RPC | 事件流、日志收集 |
| 复杂度 | 中等 | 较高 |

## 消息重试和死信队列

### RabbitMQ 死信队列

```java
@Configuration
public class DeadLetterConfig {
    
    // 业务队列
    @Bean
    public Queue businessQueue() {
        return QueueBuilder.durable("business.queue")
            .withArgument("x-dead-letter-exchange", "dead.letter.exchange")
            .withArgument("x-dead-letter-routing-key", "dead.letter")
            .build();
    }
    
    // 死信队列
    @Bean
    public Queue deadLetterQueue() {
        return new Queue("dead.letter.queue", true);
    }
    
    // 死信交换机
    @Bean
    public DirectExchange deadLetterExchange() {
        return new DirectExchange("dead.letter.exchange", true, false);
    }
    
    @Bean
    public Binding deadLetterBinding(Queue deadLetterQueue, DirectExchange deadLetterExchange) {
        return BindingBuilder.bind(deadLetterQueue)
            .to(deadLetterExchange)
            .with("dead.letter");
    }
}

@Service
@Slf4j
public class DeadLetterProcessor {
    
    @RabbitListener(queues = "dead.letter.queue")
    public void processDead Letter(Message message) {
        log.warn("Processing dead letter message: {}", new String(message.getBody()));
        // 记录到数据库、发送告警等
    }
}
```

## 消息监控

```java
@Configuration
public class KafkaMonitoringConfig {
    
    @Bean
    public MeterBinder kafkaMetrics() {
        return (registry) -> {
            // 监听消费延迟
            registry.gauge("kafka.consumer.lag", 
                () -> getConsumerLag());
        };
    }
    
    private long getConsumerLag() {
        // 计算消费者延迟
        return 0;
    }
}
```

## 总结

消息队列的关键点：

1. ✅ **RabbitMQ** - 适合任务队列和 RPC 场景
2. ✅ **Kafka** - 适合高吞吐量的事件流处理
3. ✅ **事件驱动** - 使用事件解耦业务逻辑
4. ✅ **可靠性** - 消息持久化和重试机制
5. ✅ **死信队列** - 处理失败的消息
6. ✅ **监控** - 监控消息队列的健康状态

---

这是 Spring Boot 学习文档的最后一个主题。祝你学习顺利！
