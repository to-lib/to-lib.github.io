---
sidebar_position: 9
title: "最佳实践"
description: "Kafka 生产环境最佳实践"
---

# Kafka 最佳实践

## 设计原则

### Topic 设计

#### 命名规范

```
<业务域>.<实体>.<事件类型>

示例：
- order.payment.completed
- user.registration.created
- inventory.stock.updated
```

#### 分区策略

```java
// 1. 按业务 key 分区（保证顺序）
producer.send(new ProducerRecord<>("orders", orderId, orderData));

// 2. 按时间分区（便于清理）
String key = LocalDate.now().toString();
producer.send(new ProducerRecord<>("logs", key, logData));

// 3. 轮询分区（负载均衡）
producer.send(new ProducerRecord<>("metrics", null, metricData));
```

### 副本配置

```properties
# 推荐配置
default.replication.factor=3
min.insync.replicas=2
unclean.leader.election.enable=false
```

> **说明**: 3 副本 + 2 最小同步 = 容忍 1 个节点故障

## 生产者最佳实践

### 可靠性配置

```java
Properties props = new Properties();

// 确认机制：等待所有副本确认
props.put("acks", "all");

// 幂等性：防止消息重复
props.put("enable.idempotence", "true");

// 重试配置
props.put("retries", Integer.MAX_VALUE);
props.put("delivery.timeout.ms", 120000);
props.put("max.in.flight.requests.per.connection", 5);
```

### 资源管理

```java
// 使用 try-with-resources
try (KafkaProducer<String, String> producer = new KafkaProducer<>(props)) {
    for (String message : messages) {
        producer.send(new ProducerRecord<>("topic", message));
    }
} // 自动关闭

// 或者手动管理
producer.flush();  // 确保所有消息发送
producer.close();  // 关闭连接
```

### 异常处理

```java
producer.send(record, (metadata, exception) -> {
    if (exception != null) {
        if (exception instanceof RetriableException) {
            // 可重试异常，记录后重试
            retryQueue.add(record);
        } else {
            // 不可重试异常，记录到死信队列
            deadLetterQueue.add(record);
        }
        logger.error("发送失败", exception);
    }
});
```

## 消费者最佳实践

### 位移管理

```java
// 推荐：手动提交
props.put("enable.auto.commit", "false");

try {
    while (running) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));

        for (ConsumerRecord<String, String> record : records) {
            processRecord(record);  // 先处理
        }

        consumer.commitSync();  // 再提交
    }
} finally {
    consumer.commitSync();  // 关闭前提交
    consumer.close();
}
```

### 优雅关闭

```java
private volatile boolean running = true;

public void shutdown() {
    running = false;
    consumer.wakeup();  // 唤醒阻塞的 poll()
}

// 在主循环中
try {
    while (running) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
        // 处理消息
    }
} catch (WakeupException e) {
    if (running) throw e;  // 意外唤醒
} finally {
    consumer.commitSync();
    consumer.close();
}
```

### 消息处理

```java
// 幂等处理
public void processRecord(ConsumerRecord<String, String> record) {
    String messageId = record.headers().lastHeader("message-id").value().toString();

    // 检查是否已处理
    if (processedMessages.contains(messageId)) {
        logger.info("消息已处理，跳过: {}", messageId);
        return;
    }

    // 处理消息
    doProcess(record);

    // 记录已处理
    processedMessages.add(messageId);
}
```

## 运维最佳实践

### 监控指标

```java
// 生产者监控
record-send-rate          // 发送速率
record-error-rate         // 错误率
request-latency-avg       // 平均延迟
batch-size-avg            // 平均批次大小

// 消费者监控
records-consumed-rate     // 消费速率
records-lag-max           // 最大积压
fetch-latency-avg         // 拉取延迟
commit-latency-avg        // 提交延迟

// Broker 监控
UnderReplicatedPartitions // 欠复制分区
OfflinePartitionsCount    // 离线分区
ActiveControllerCount     // 活跃控制器
```

### 告警规则

```yaml
# Prometheus 告警规则示例
groups:
  - name: kafka
    rules:
      - alert: KafkaConsumerLag
        expr: kafka_consumergroup_lag > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "消费者积压超过 10000"

      - alert: KafkaUnderReplicated
        expr: kafka_server_replicamanager_underreplicatedpartitions > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "存在欠复制分区"
```

### 日志配置

```properties
# log4j.properties
log4j.rootLogger=INFO, stdout, kafkaAppender

log4j.appender.kafkaAppender=org.apache.kafka.log4jappender.KafkaLog4jAppender
log4j.appender.kafkaAppender.brokerList=localhost:9092
log4j.appender.kafkaAppender.topic=application-logs
```

## 安全最佳实践

### 认证配置

```properties
# 生产者/消费者
security.protocol=SASL_SSL
sasl.mechanism=SCRAM-SHA-512
sasl.jaas.config=org.apache.kafka.common.security.scram.ScramLoginModule required \
  username="user" \
  password="password";
```

### 加密传输

```properties
# SSL 配置
ssl.keystore.location=/path/to/keystore.jks
ssl.keystore.password=keystore-password
ssl.key.password=key-password
ssl.truststore.location=/path/to/truststore.jks
ssl.truststore.password=truststore-password
```

### 权限控制

```bash
# 最小权限原则
# 生产者只允许写入
kafka-acls.sh --add --allow-principal User:producer \
  --operation Write --topic orders

# 消费者只允许读取
kafka-acls.sh --add --allow-principal User:consumer \
  --operation Read --topic orders \
  --group order-processor
```

## 错误处理模式

### 死信队列

```java
public void consumeWithDLQ() {
    while (running) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));

        for (ConsumerRecord<String, String> record : records) {
            try {
                processRecord(record);
            } catch (Exception e) {
                // 发送到死信队列
                sendToDLQ(record, e);
            }
        }

        consumer.commitSync();
    }
}

private void sendToDLQ(ConsumerRecord<String, String> record, Exception e) {
    ProducerRecord<String, String> dlqRecord = new ProducerRecord<>(
        record.topic() + ".dlq",
        record.key(),
        record.value()
    );
    dlqRecord.headers()
        .add("original-topic", record.topic().getBytes())
        .add("error-message", e.getMessage().getBytes());

    dlqProducer.send(dlqRecord);
}
```

### 重试策略

```java
public void processWithRetry(ConsumerRecord<String, String> record) {
    int maxRetries = 3;
    int retryCount = 0;

    while (retryCount < maxRetries) {
        try {
            processRecord(record);
            return;
        } catch (RetriableException e) {
            retryCount++;
            long backoff = (long) Math.pow(2, retryCount) * 100;
            Thread.sleep(backoff);
        }
    }

    // 达到最大重试次数，发送到 DLQ
    sendToDLQ(record, new MaxRetriesExceededException());
}
```

## 测试最佳实践

### 单元测试

```java
// 使用 MockProducer
MockProducer<String, String> mockProducer = new MockProducer<>(
    true, new StringSerializer(), new StringSerializer());

// 测试发送
myService.sendMessage("test-message");
assertEquals(1, mockProducer.history().size());
assertEquals("test-message", mockProducer.history().get(0).value());
```

### 集成测试

```java
// 使用 Testcontainers
@Container
static KafkaContainer kafka = new KafkaContainer(
    DockerImageName.parse("confluentinc/cp-kafka:7.5.0"));

@Test
void testProducerConsumer() {
    Properties props = new Properties();
    props.put("bootstrap.servers", kafka.getBootstrapServers());
    // ... 测试代码
}
```

## 检查清单

### 部署前检查

- [ ] Topic 分区数和副本数配置正确
- [ ] 生产者配置了 `acks=all` 和幂等性
- [ ] 消费者配置了手动提交
- [ ] 配置了监控和告警
- [ ] 配置了访问控制和加密
- [ ] 进行了性能测试
- [ ] 准备了故障恢复方案

### 运维检查

- [ ] 定期检查消费者积压
- [ ] 监控 Broker 磁盘使用率
- [ ] 检查欠复制分区
- [ ] 定期备份 Topic 配置
- [ ] 制定日志清理策略

## 下一步

- ⚡ [性能优化](/docs/kafka/performance-optimization) - 性能调优指南
- 🔧 [集群管理](/docs/kafka/cluster-management) - 集群管理操作
- 📊 [监控与运维](/docs/kafka/monitoring) - 监控告警配置

## 参考资料

- [Confluent 最佳实践](https://docs.confluent.io/platform/current/kafka/deployment.html)
- [Kafka 官方文档](https://kafka.apache.org/documentation/)
