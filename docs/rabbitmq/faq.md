---
sidebar_position: 14
title: "常见问题"
description: "RabbitMQ 常见问题解答"
---

# RabbitMQ 常见问题

## 连接问题

### Q: 无法连接到 RabbitMQ？

**A:** 检查以下几点：

```bash
# 1. 检查服务是否运行
rabbitmqctl status

# 2. 检查端口是否开放
nc -zv localhost 5672

# 3. 检查防火墙
sudo iptables -L | grep 5672

# 4. 检查用户权限
rabbitmqctl list_permissions
```

### Q: 连接频繁断开？

**A:** 可能原因和解决方案：

1. **心跳超时**：调整心跳间隔

   ```java
   factory.setRequestedHeartbeat(60);
   ```

2. **网络问题**：检查网络稳定性

3. **启用自动恢复**：
   ```java
   factory.setAutomaticRecoveryEnabled(true);
   factory.setNetworkRecoveryInterval(5000);
   ```

### Q: 连接数过多导致性能下降？

**A:** 优化连接策略：

1. **使用连接池**：复用连接
2. **使用多个 Channel**：而不是多个连接
3. **调整连接限制**：
   ```ini
   # rabbitmq.conf
   connection_max = 10000
   channel_max = 2047
   ```

## 消息问题

### Q: 消息丢失怎么办？

**A:** 确保以下配置：

1. **队列持久化**：`durable = true`
2. **消息持久化**：`deliveryMode = 2`
3. **禁用自动确认**：`autoAck = false`
4. **启用发布确认**：`confirmSelect()`

```java
// 完整示例
channel.queueDeclare("queue", true, false, false, null);  // 持久化队列
channel.confirmSelect();  // 开启发布确认

channel.basicPublish("", "queue",
    MessageProperties.PERSISTENT_TEXT_PLAIN,  // 持久化消息
    message.getBytes());

channel.waitForConfirms(5000);  // 等待确认
```

### Q: 消息重复消费？

**A:** 实现幂等性：

```java
// 方案1: 使用 Redis 去重
String messageId = properties.getMessageId();
if (!redis.setIfAbsent("processed:" + messageId, "1", 24, TimeUnit.HOURS)) {
    return; // 已处理过
}
processMessage(body);

// 方案2: 数据库唯一约束
try {
    messageLogRepository.save(new MessageLog(messageId));
    processMessage(body);
} catch (DuplicateKeyException e) {
    log.info("消息已处理: {}", messageId);
}
```

### Q: 消息顺序无法保证？

**A:** RabbitMQ 只保证队列内有序：

1. **单一队列单一消费者**：保证顺序
2. **业务分区**：将相关消息发送到同一队列
3. **消息序号**：携带序号在消费端排序

### Q: 消息积压严重？

**A:** 解决方案：

1. **增加消费者数量**
2. **提高预取值**：`channel.basicQos(100)`
3. **优化消费逻辑**：异步处理
4. **临时清理**：
   ```bash
   rabbitmqctl purge_queue queue_name
   ```
5. **设置 TTL**：让过期消息自动进入死信

## 队列问题

### Q: 队列消息积压导致内存溢出？

**A:** 多种解决方案：

```java
// 1. 使用 Lazy Queue
Map<String, Object> args = new HashMap<>();
args.put("x-queue-mode", "lazy");
channel.queueDeclare("lazy_queue", true, false, false, args);

// 2. 设置队列长度限制
args.put("x-max-length", 10000);
args.put("x-overflow", "reject-publish");  // 或 "drop-head"
```

```ini
# 3. 调整内存阈值
vm_memory_high_watermark.relative = 0.4
```

### Q: 队列无法删除？

**A:** 确保没有消费者：

```bash
# 查看队列消费者
rabbitmqctl list_queues name consumers

# 强制删除
rabbitmqctl delete_queue queue_name
```

### Q: 如何选择 Classic、Quorum 还是 Stream 队列？

**A:** 根据场景选择：

| 队列类型 | 适用场景       | 特点                     |
| -------- | -------------- | ------------------------ |
| Classic  | 普通场景       | 性能较高，旧版默认       |
| Quorum   | 高可靠性需求   | 强一致性，推荐新项目     |
| Stream   | 大数据量、回溯 | 类似 Kafka，支持重复消费 |

## Spring Boot 集成问题

### Q: Spring Boot 如何配置手动确认？

**A:** 配置和代码：

```yaml
spring:
  rabbitmq:
    listener:
      simple:
        acknowledge-mode: manual
        prefetch: 10
```

```java
@RabbitListener(queues = "myQueue")
public void receive(Message message, Channel channel,
                    @Header(AmqpHeaders.DELIVERY_TAG) long tag) throws IOException {
    try {
        processMessage(message);
        channel.basicAck(tag, false);
    } catch (Exception e) {
        channel.basicNack(tag, false, true);
    }
}
```

### Q: Spring Boot 如何发送延迟消息？

**A:** 使用延迟消息插件：

```java
@Bean
public CustomExchange delayedExchange() {
    Map<String, Object> args = new HashMap<>();
    args.put("x-delayed-type", "direct");
    return new CustomExchange("delayed.exchange", "x-delayed-message", true, false, args);
}

// 发送
public void sendDelayed(String message, int delayMs) {
    rabbitTemplate.convertAndSend("delayed.exchange", "key", message, msg -> {
        msg.getMessageProperties().setDelay(delayMs);
        return msg;
    });
}
```

### Q: 消息序列化失败？

**A:** 配置 JSON 序列化：

```java
@Bean
public MessageConverter jsonMessageConverter() {
    return new Jackson2JsonMessageConverter();
}

@Bean
public RabbitTemplate rabbitTemplate(ConnectionFactory cf) {
    RabbitTemplate template = new RabbitTemplate(cf);
    template.setMessageConverter(jsonMessageConverter());
    return template;
}
```

### Q: 如何实现消息重试？

**A:** 配置重试策略：

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

## 安全问题

### Q: 如何修改默认的 guest 用户？

**A:** 出于安全考虑：

```bash
# 创建新管理员
rabbitmqctl add_user admin SecurePass123!
rabbitmqctl set_user_tags admin administrator
rabbitmqctl set_permissions -p / admin ".*" ".*" ".*"

# 删除 guest
rabbitmqctl delete_user guest
```

### Q: 如何启用 SSL/TLS？

**A:** 配置步骤：

```ini
# rabbitmq.conf
listeners.ssl.default = 5671

ssl_options.cacertfile = /path/to/ca_cert.pem
ssl_options.certfile   = /path/to/server_cert.pem
ssl_options.keyfile    = /path/to/server_key.pem
ssl_options.verify     = verify_peer
ssl_options.fail_if_no_peer_cert = true
```

### Q: 如何限制用户只能访问特定队列？

**A:** 使用正则权限：

```bash
# 只允许访问以 "app." 开头的资源
rabbitmqctl set_permissions -p / app_user "^app\\..*" "^app\\..*" "^app\\..*"
```

## 集群问题

### Q: 节点无法加入集群？

**A:** 检查：

1. **Erlang Cookie 一致**

   ```bash
   cat /var/lib/rabbitmq/.erlang.cookie
   ```

2. **主机名解析正确**

   ```bash
   ping rabbit@node1
   ```

3. **端口开放**：4369, 25672

### Q: 网络分区如何处理？

**A:** 配置分区处理策略：

```ini
# rabbitmq.conf
cluster_partition_handling = autoheal
# 可选值: ignore, pause_minority, autoheal
```

### Q: 镜像队列和 Quorum 队列如何选择？

**A:**

- **Quorum 队列（推荐）**：RabbitMQ 3.8+，强一致性，更好的数据安全
- **镜像队列**：旧版兼容，将逐步废弃

```bash
# 设置 Quorum 队列
rabbitmqctl set_policy quorum ".*" '{"queue-mode":"quorum"}' --apply-to queues
```

## 性能问题

### Q: 吞吐量低怎么办？

**A:** 优化建议：

1. **使用批量发送和确认**
2. **调整 QoS 预取值**：`basicQos(100)`
3. **使用异步确认**
4. **减少消息大小**
5. **使用多个 Channel**

### Q: 内存使用过高？

**A:** 解决方案：

```ini
# rabbitmq.conf
vm_memory_high_watermark.relative = 0.4
disk_free_limit.relative = 2.0
```

```java
// 使用 Lazy Queue
args.put("x-queue-mode", "lazy");
```

### Q: 如何监控 RabbitMQ 性能？

**A:** 多种方式：

1. **Management UI**：http://localhost:15672
2. **Prometheus + Grafana**：
   ```bash
   rabbitmq-plugins enable rabbitmq_prometheus
   ```
3. **命令行**：
   ```bash
   rabbitmqctl list_queues name messages consumers memory
   ```

## 版本升级

### Q: 如何升级 RabbitMQ？

**A:** 升级步骤：

1. **备份数据**：

   ```bash
   rabbitmqctl export_definitions definitions.json
   ```

2. **查看兼容性**：检查 Erlang 版本要求

3. **滚动升级**（集群）：

   - 一次升级一个节点
   - 确认节点恢复后再升级下一个

4. **恢复数据**：
   ```bash
   rabbitmqctl import_definitions definitions.json
   ```

### Q: 如何从 3.8 升级到 3.12？

**A:** 注意事项：

1. 先升级到最新的 3.8.x
2. 检查弃用的功能（如镜像队列策略）
3. 更新客户端库版本
4. 测试应用兼容性

## 常用命令速查

```bash
# 服务管理
rabbitmqctl status
rabbitmqctl stop_app
rabbitmqctl start_app

# 队列管理
rabbitmqctl list_queues
rabbitmqctl purge_queue <queue>
rabbitmqctl delete_queue <queue>

# 用户管理
rabbitmqctl add_user <user> <pass>
rabbitmqctl set_permissions -p / <user> ".*" ".*" ".*"

# 插件管理
rabbitmq-plugins enable rabbitmq_management
rabbitmq-plugins list
```

## 参考资料

- [RabbitMQ 故障排查](https://www.rabbitmq.com/troubleshooting.html)
- [RabbitMQ 升级指南](https://www.rabbitmq.com/upgrade.html)
- [Spring AMQP 文档](https://docs.spring.io/spring-amqp/docs/current/reference/html/)
