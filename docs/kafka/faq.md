---
sidebar_position: 11
title: "常见问题"
description: "Kafka 使用中的常见问题解答"
---

# Kafka 常见问题

## 基础概念

### Q: Kafka 和传统消息队列有什么区别？

**A:** 主要区别：

- **消息保留**：Kafka 持久化所有消息，可重复消费；传统 MQ 消费后删除
- **吞吐量**：Kafka 支持更高的吞吐量（TB/s 级别）
- **消费模型**：Kafka 支持多消费者组独立消费
- **顺序保证**：Kafka 保证分区内有序
- **消息回溯**：Kafka 支持消费历史消息，传统 MQ 不支持

### Q: 什么时候使用 Kafka？

**A:** 适合场景：

- 高吞吐量消息传递
- 实时流处理
- 日志聚合
- 事件溯源
- 微服务异步通信
- 数据管道和 ETL

### Q: Kafka 的消息可以被多次消费吗？

**A:** 是的。Kafka 消息在保留期内可以被多次消费：

- 不同消费者组可以独立消费相同的消息
- 同一消费者组可以通过重置 offset 重新消费
- 消息保留策略可以基于时间或大小配置

## 性能问题

### Q: 如何提高 Kafka 生产者性能？

**A:** 优化建议：

```java
// 增大批次大小
props.put("batch.size", 32768);
// 增加等待时间允许更多消息累积
props.put("linger.ms", 20);
// 启用压缩
props.put("compression.type", "lz4");
// 增大缓冲区
props.put("buffer.memory", 67108864);
// 使用异步发送
producer.send(record, callback);
```

### Q: 消费者拉取消息很慢怎么办？

**A:** 解决方法：

1. **增加消费者数量**（不超过分区数）
2. **调整拉取参数**：
   ```java
   props.put("max.poll.records", 1000);
   props.put("fetch.min.bytes", 1024);
   props.put("fetch.max.wait.ms", 500);
   ```
3. **多线程处理消息**（注意线程安全）
4. **优化消息处理逻辑**
5. **检查网络带宽**

### Q: 为什么生产者发送延迟很高？

**A:** 常见原因及解决：

| 原因                    | 解决方案               |
| ----------------------- | ---------------------- |
| `linger.ms` 设置过大    | 减小该值               |
| `acks=all` 等待副本同步 | 评估是否需要降低可靠性 |
| 网络延迟高              | 检查网络配置           |
| Broker 过载             | 增加 Broker 或分区     |
| 未开启压缩导致数据量大  | 开启 lz4 压缩          |

## 可靠性问题

### Q: 如何防止消息丢失？

**A:** 三端配置保障：

```java
// 生产者端
props.put("acks", "all");                    // 等待所有副本确认
props.put("retries", Integer.MAX_VALUE);     // 无限重试
props.put("enable.idempotence", "true");     // 开启幂等性
props.put("max.in.flight.requests.per.connection", 5); // 幂等性下可>1

// Broker 端
// server.properties
min.insync.replicas=2
unclean.leader.election.enable=false

// 消费者端
props.put("enable.auto.commit", "false");
// 处理完消息后手动提交
consumer.commitSync();
```

### Q: 如何避免消息重复？

**A:** 解决方案：

1. **生产者幂等性**：`enable.idempotence=true`
2. **事务消息**：使用 Kafka 事务 API
3. **消费端去重**：
   - 数据库唯一约束
   - Redis 去重
   - 业务层判断

### Q: acks 配置有什么区别？

**A:** 三种配置对比：

| acks  | 说明                 | 可靠性 | 性能 |
| ----- | -------------------- | ------ | ---- |
| `0`   | 不等待确认，发送即忘 | 低     | 最高 |
| `1`   | Leader 写入即返回    | 中     | 高   |
| `all` | 等待 ISR 全部写入    | 高     | 较低 |

## 运维问题

### Q: 如何监控 Kafka 集群？

**A:** 推荐监控方案：

1. **JMX 指标**：原生支持，适合自建监控
2. **Prometheus + Grafana**：云原生首选
3. **Kafka Manager / CMAK**：可视化管理工具
4. **Confluent Control Center**：企业级方案
5. **Burrow**：专注消费者 lag 监控

### Q: Topic 分区数如何选择？

**A:** 计算公式：

```
分区数 = max(目标吞吐量/生产者单分区吞吐量, 目标吞吐量/消费者单分区吞吐量)
```

经验建议：

- 一般建议每个 Broker 承载 100-1000 个分区
- 分区数一旦增加不能减少
- 分区过多会导致 Controller 压力增大

### Q: 如何扩展 Kafka 集群？

**A:** 扩展步骤：

1. **添加新 Broker**
2. **生成分区重分配计划**：
   ```bash
   kafka-reassign-partitions.sh --generate \
     --bootstrap-server localhost:9092 \
     --topics-to-move-json-file topics.json \
     --broker-list "1,2,3,4"
   ```
3. **执行重分配**：
   ```bash
   kafka-reassign-partitions.sh --execute \
     --bootstrap-server localhost:9092 \
     --reassignment-json-file plan.json
   ```
4. **验证重分配完成**

## 故障排查

### Q: 消费者频繁发生再均衡怎么办？

**A:** 排查和解决：

```java
// 增大会话超时
props.put("session.timeout.ms", 45000);
// 增大心跳间隔
props.put("heartbeat.interval.ms", 15000);
// 增大处理超时
props.put("max.poll.interval.ms", 600000);
// 减少单次拉取数量
props.put("max.poll.records", 100);

// 使用静态成员资格（Kafka 2.3+）
props.put("group.instance.id", "consumer-1");
```

### Q: 连接 Kafka 超时怎么办？

**A:** 排查步骤：

1. **检查网络连通性**：`telnet kafka-host 9092`
2. **检查 `advertised.listeners`** 配置是否正确
3. **检查防火墙规则**
4. **检查 DNS 解析**
5. **检查 Broker 是否启动**：`jps | grep Kafka`

### Q: 出现 "Not enough replicas" 错误？

**A:** 原因和解决：

- **原因**：可用 ISR 副本数小于 `min.insync.replicas`
- **排查**：
  ```bash
  # 查看 Topic ISR
  kafka-topics.sh --describe --topic my-topic \
    --bootstrap-server localhost:9092
  ```
- **解决**：
  1. 检查 Broker 是否正常运行
  2. 临时降低 `min.insync.replicas`
  3. 修复故障 Broker

### Q: 消息堆积严重怎么处理？

**A:** 处理方案：

1. **临时增加消费者实例**
2. **跳过部分消息**（若业务允许）：
   ```bash
   kafka-consumer-groups.sh --reset-offsets --to-latest \
     --group my-group --topic my-topic \
     --bootstrap-server localhost:9092 --execute
   ```
3. **增加分区数**（需重新消费者分配）
4. **优化消费者处理逻辑**

## 版本升级

### Q: 如何进行 Kafka 滚动升级？

**A:** 升级步骤：

1. **更新配置**：设置 `inter.broker.protocol.version` 为当前版本
2. **逐个升级 Broker**：停止 → 更新 → 启动
3. **验证集群稳定**
4. **更新协议版本**为新版本
5. **滚动重启**所有 Broker

### Q: 升级到 Kafka 3.x 需要注意什么？

**A:** 关键变化：

- KRaft 模式可选（移除 ZooKeeper 依赖）
- 部分配置参数变更
- 一些废弃 API 移除
- 建议先在测试环境验证

## Spring Boot 集成

### Q: Spring Boot 如何配置 Kafka？

**A:** 基本配置：

```yaml
# application.yml
spring:
  kafka:
    bootstrap-servers: localhost:9092
    producer:
      key-serializer: org.apache.kafka.common.serialization.StringSerializer
      value-serializer: org.apache.kafka.common.serialization.StringSerializer
      acks: all
    consumer:
      group-id: my-group
      auto-offset-reset: earliest
      enable-auto-commit: false
```

```java
// 生产者
@Autowired
private KafkaTemplate<String, String> kafkaTemplate;

public void send(String message) {
    kafkaTemplate.send("topic", message);
}

// 消费者
@KafkaListener(topics = "topic", groupId = "my-group")
public void listen(String message) {
    System.out.println("Received: " + message);
}
```

### Q: Spring Kafka 消费者如何手动提交？

**A:** 配置和代码：

```yaml
spring:
  kafka:
    consumer:
      enable-auto-commit: false
    listener:
      ack-mode: manual
```

```java
@KafkaListener(topics = "topic")
public void listen(String message, Acknowledgment ack) {
    // 处理消息
    process(message);
    // 手动确认
    ack.acknowledge();
}
```

## 参考资料

- [Kafka FAQ 官方文档](https://kafka.apache.org/documentation/#faq)
- [Kafka 故障排查指南](https://kafka.apache.org/documentation/#operations)
