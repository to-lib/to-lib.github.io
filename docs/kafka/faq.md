---
sidebar_position: 11
title: "常见问题"
description: "Kafka 使用中的常见问题解答"
---

# Kafka 常见问题

## 基础问题

### Q: Kafka 和传统消息队列有什么区别？

**A:** 主要区别：

- **消息保留**：Kafka 持久化所有消息，可重复消费；传统 MQ 消费后删除
- **吞吐量**：Kafka 支持更高的吞吐量（TB/s 级别）
- **消费模型**：Kafka 支持多消费者组独立消费
- **顺序保证**：Kafka 保证分区内有序

### Q: 什么时候使用 Kafka？

**A:** 适合场景：

- 高吞吐量消息传递
- 实时流处理
- 日志聚合
- 事件溯源
- 微服务异步通信

## 性能问题

### Q: 如何提高 Kafka 生产者性能？

**A:** 优化建议：

```java
props.put("batch.size", 32768);
props.put("linger.ms", 20);
props.put("compression.type", "lz4");
props.put("buffer.memory", 67108864);
```

### Q: 消费者拉取消息很慢怎么办？

**A:** 解决方法：

1. 增加消费者数量（不超过分区数）
2. 增加 `max.poll.records`
3. 使用多线程处理消息
4. 优化消息处理逻辑

## 可靠性问题

### Q: 如何防止消息丢失？

**A:** 配置建议：

```java
// 生产者
props.put("acks", "all");
props.put("retries", Integer.MAX_VALUE);
props.put("enable.idempotence", "true");

// 消费者
props.put("enable.auto.commit", "false");
// 手动提交位移
consumer.commitSync();
```

### Q: 如何避免消息重复？

**A:** 解决方案：

1. 开启生产者幂等性
2. 使用事务
3. 消费端去重（业务层实现）

## 运维问题

### Q: 如何监控 Kafka 集群？

**A:** 监控方案：

- JMX 指标
- Kafka Manager
- Confluent Control Center
- Prometheus + Grafana

### Q: Topic 分区数如何选择？

**A:** 计算公式：

```
分区数 = max(目标吞吐量/生产者单分区吞吐量, 目标吞吐量/消费者单分区吞吐量)
```

## 参考资料

- [Kafka FAQ 官方文档](https://kafka.apache.org/documentation/#faq)
