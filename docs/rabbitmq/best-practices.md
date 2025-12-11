---
sidebar_position: 9
title: "最佳实践"
description: "RabbitMQ 生产环境最佳实践"
---

# RabbitMQ 最佳实践

本指南总结 RabbitMQ 在生产环境中的最佳实践。

## 连接管理

### ✅ 复用连接

```java
// 好：复用连接
public class ConnectionManager {
    private static Connection connection;

    public static synchronized Connection getConnection() throws Exception {
        if (connection == null || !connection.isOpen()) {
            ConnectionFactory factory = new ConnectionFactory();
            factory.setHost("localhost");
            factory.setAutomaticRecoveryEnabled(true);
            connection = factory.newConnection();
        }
        return connection;
    }
}
```

### ❌ 避免频繁创建

```java
// 差：每次发送都创建新连接
public void send(String message) {
    Connection connection = factory.newConnection(); // 避免！
    Channel channel = connection.createChannel();
    channel.basicPublish("", "queue", null, message.getBytes());
    connection.close();
}
```

## 通道管理

### 线程安全

```java
// 使用 ThreadLocal 管理通道
private static final ThreadLocal<Channel> channelHolder = ThreadLocal.withInitial(() -> {
    try {
        return ConnectionManager.getConnection().createChannel();
    } catch (Exception e) {
        throw new RuntimeException(e);
    }
});

public static Channel getChannel() {
    return channelHolder.get();
}
```

## 消息可靠性

### 生产者端

```java
public class ReliableProducer {
    private final Channel channel;

    public void send(String message) throws Exception {
        // 1. 启用发布确认
        channel.confirmSelect();

        // 2. 持久化消息
        AMQP.BasicProperties props = MessageProperties.PERSISTENT_TEXT_PLAIN;

        // 3. 设置 mandatory
        channel.basicPublish("exchange", "key", true, props, message.getBytes());

        // 4. 等待确认
        if (!channel.waitForConfirms(5000)) {
            throw new RuntimeException("消息未被确认");
        }
    }
}
```

### 消费者端

```java
public class ReliableConsumer {

    public void consume(Channel channel) throws Exception {
        // 1. 设置 QoS
        channel.basicQos(10);

        // 2. 手动确认
        boolean autoAck = false;

        DeliverCallback callback = (consumerTag, delivery) -> {
            try {
                processMessage(delivery.getBody());
                // 3. 成功后确认
                channel.basicAck(delivery.getEnvelope().getDeliveryTag(), false);
            } catch (Exception e) {
                // 4. 失败重新入队或进入死信
                channel.basicNack(delivery.getEnvelope().getDeliveryTag(), false, !isRetryExhausted());
            }
        };

        channel.basicConsume("queue", autoAck, callback, consumerTag -> {});
    }
}
```

## 幂等性处理

```java
@Service
public class IdempotentHandler {

    @Autowired
    private RedisTemplate<String, String> redis;

    public boolean processIfNew(String messageId, Runnable task) {
        String key = "msg:processed:" + messageId;

        // 使用 Redis SETNX 实现幂等
        Boolean isNew = redis.opsForValue().setIfAbsent(key, "1", 24, TimeUnit.HOURS);

        if (Boolean.TRUE.equals(isNew)) {
            try {
                task.run();
                return true;
            } catch (Exception e) {
                redis.delete(key); // 失败时删除，允许重试
                throw e;
            }
        }
        return false;
    }
}
```

## 死信队列

### 配置

```java
public void setupDeadLetterQueue(Channel channel) throws Exception {
    // 死信交换机和队列
    channel.exchangeDeclare("dlx", "direct", true);
    channel.queueDeclare("dead-letter-queue", true, false, false, null);
    channel.queueBind("dead-letter-queue", "dlx", "dead");

    // 业务队列配置死信
    Map<String, Object> args = new HashMap<>();
    args.put("x-dead-letter-exchange", "dlx");
    args.put("x-dead-letter-routing-key", "dead");

    channel.queueDeclare("business-queue", true, false, false, args);
}
```

### 死信消费者

```java
@RabbitListener(queues = "dead-letter-queue")
public void handleDeadLetter(Message message) {
    // 记录日志
    log.error("死信消息: {}", new String(message.getBody()));

    // 告警通知
    alertService.send("收到死信消息: " + message.getMessageProperties().getMessageId());

    // 保存到数据库待人工处理
    deadLetterRepository.save(message);
}
```

## 延迟消息

### 使用插件

```bash
# 启用延迟消息插件
rabbitmq-plugins enable rabbitmq_delayed_message_exchange
```

```java
// 声明延迟交换机
Map<String, Object> args = new HashMap<>();
args.put("x-delayed-type", "direct");

channel.exchangeDeclare("delayed-exchange", "x-delayed-message", true, false, args);

// 发送延迟消息
AMQP.BasicProperties props = new AMQP.BasicProperties.Builder()
    .headers(Map.of("x-delay", 60000)) // 延迟 60 秒
    .build();

channel.basicPublish("delayed-exchange", "key", props, message.getBytes());
```

### 使用 TTL + 死信

```java
// 延迟队列
Map<String, Object> args = new HashMap<>();
args.put("x-message-ttl", 60000);
args.put("x-dead-letter-exchange", "target-exchange");
args.put("x-dead-letter-routing-key", "target-key");

channel.queueDeclare("delay-60s", true, false, false, args);
```

## 监控告警

### 关键指标

```java
@Scheduled(fixedRate = 60000)
public void monitorQueues() {
    // 检查队列深度
    int depth = getQueueDepth("important-queue");
    if (depth > 10000) {
        alert("队列积压告警", "important-queue 深度: " + depth);
    }

    // 检查消费者数量
    int consumers = getConsumerCount("important-queue");
    if (consumers == 0) {
        alert("消费者离线", "important-queue 没有消费者");
    }
}
```

### Prometheus 指标

```yaml
# prometheus.yml
scrape_configs:
  - job_name: "rabbitmq"
    static_configs:
      - targets: ["localhost:15692"]
```

## 安全配置

### 用户权限

```bash
# 创建用户
rabbitmqctl add_user app_user strong_password

# 设置权限（配置/写/读）
rabbitmqctl set_permissions -p /app app_user "^app\." "^app\." "^app\."

# 设置用户标签
rabbitmqctl set_user_tags app_user monitoring
```

### SSL/TLS

```ini
# rabbitmq.conf
listeners.ssl.default = 5671
ssl_options.cacertfile = /path/to/ca_certificate.pem
ssl_options.certfile = /path/to/server_certificate.pem
ssl_options.keyfile = /path/to/server_key.pem
ssl_options.verify = verify_peer
ssl_options.fail_if_no_peer_cert = true
```

## 生产检查清单

### 部署前

- [ ] 至少 3 个节点的集群
- [ ] 启用持久化
- [ ] 配置镜像队列或 Quorum 队列
- [ ] 设置内存和磁盘告警阈值
- [ ] 配置监控和告警
- [ ] 备份策略

### 应用层

- [ ] 连接自动恢复
- [ ] 发布确认
- [ ] 消费者手动确认
- [ ] 幂等处理
- [ ] 死信队列
- [ ] 重试机制

### 运维

- [ ] 日志收集
- [ ] 指标监控
- [ ] 告警配置
- [ ] 定期备份
- [ ] 容量规划

## 常见错误

### 1. 内存耗尽

```bash
# 检查内存使用
rabbitmqctl status | grep memory

# 设置内存限制
# rabbitmq.conf
vm_memory_high_watermark.relative = 0.4
```

### 2. 磁盘空间不足

```bash
# 设置磁盘限制
# rabbitmq.conf
disk_free_limit.absolute = 5GB
```

### 3. 连接数过多

```bash
# 检查连接
rabbitmqctl list_connections

# 设置连接限制
# rabbitmq.conf
channel_max = 2047
```

## 下一步

- 📊 [监控运维](/docs/rabbitmq/monitoring) - 监控 RabbitMQ
- ❓ [常见问题](/docs/rabbitmq/faq) - FAQ
- 💼 [面试题集](/docs/rabbitmq/interview-questions) - 面试常见问题

## 参考资料

- [RabbitMQ 生产检查清单](https://www.rabbitmq.com/production-checklist.html)
- [可靠性指南](https://www.rabbitmq.com/reliability.html)
