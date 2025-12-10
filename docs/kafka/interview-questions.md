---
sidebar_position: 12
title: "面试题集"
description: "Kafka 常见面试题及答案"
---

# Kafka 面试题集

## 基础概念

### 1. 什么是 Kafka？它的主要应用场景有哪些？

**答案：**
Kafka 是一个分布式流处理平台，主要用于：

- 消息队列
- 日志聚合
- 流处理
- 事件溯源
- 指标监控

### 2. Kafka 的核心组件有哪些？

**答案：**

- **Producer**：生产者，发送消息
- **Consumer**：消费者，消费消息
- **Broker**：服务器节点
- **Topic**：消息分类
- **Partition**：分区，实现并行
- **ZooKeeper/KRaft**：集群协调

## 架构设计

### 3. Kafka 如何保证消息不丢失？

**答案：**

1. **生产者端**：

   - `acks=all`：等待所有副本确认
   - 开启幂等性：`enable.idempotence=true`
   - 配置重试：`retries=Integer.MAX_VALUE`

2. **Broker 端**：

   - 配置副本数：`replication.factor>=3`
   - 配置 ISR：`min.insync.replicas>=2`

3. **消费者端**：
   - 手动提交位移
   - 确保消息处理完成后再提交

### 4. Kafka 如何保证消息的顺序性？

**答案：**

- **分区内有序**：同一分区内消息严格有序
- **全局有序**：设置 Topic 只有 1 个分区
- **业务有序**：使用相同的 key 发送到同一分区

```java
// 确保相同用户的消息有序
producer.send(new ProducerRecord<>("topic", userId, message));
```

### 5. 什么是 ISR？

**答案：**
ISR（In-Sync Replicas）是同步副本集合，包含：

- Leader 副本
- 与 Leader 保持同步的 Follower 副本

只有 ISR 中的副本才有资格被选举为 Leader。

## 性能优化

### 6. Kafka 为什么这么快？

**答案：**

1. **顺序写磁盘**：比随机写内存还快
2. **零拷贝**：减少数据复制
3. **批量处理**：批量发送和消费
4. **分区并行**：提高吞吐量
5. **页缓存**：利用操作系统缓存

### 7. 如何优化 Kafka 生产者性能？

**答案：**

```java
// 增大批次
props.put("batch.size", 32768);
// 增加等待时间
props.put("linger.ms", 20);
// 启用压缩
props.put("compression.type", "lz4");
// 增大缓冲区
props.put("buffer.memory", 67108864);
```

## 高级特性

### 8. Kafka 事务是如何实现的？

**答案：**
Kafka 通过以下机制实现事务：

1. **幂等性**：保证消息不重复
2. **事务 ID**：标识事务
3. **事务协调器**：管理事务状态
4. **两阶段提交**：确保原子性

```java
producer.initTransactions();
producer.beginTransaction();
producer.send(record);
producer.commitTransaction();
```

### 9. 什么是 Exactly-Once 语义？

**答案：**
精确一次语义保证消息既不丢失也不重复。实现方式：

- **生产者**：启用幂等性和事务
- **消费者**：设置隔离级别为 `read_committed`
- **流处理**：使用 Kafka Streams 的事务 API

## 运维相关

### 10. Kafka 如何进行扩容？

**答案：**

1. **增加 Broker**：添加新节点到集群
2. **增加分区**：分配分区到新节点
3. **数据迁移**：使用分区重分配工具

```bash
kafka-reassign-partitions.sh --execute \
  --bootstrap-server localhost:9092 \
  --reassignment-json-file reassign.json
```

### 11. 如何处理消费者组的再均衡？

**答案：**
减少再均衡的方法：

- 增加 `session.timeout.ms`
- 增加 `max.poll.interval.ms`
- 减少 `max.poll.records`
- 使用静态成员资格（Kafka 2.3+）

### 12. Kafka 丢消息的场景有哪些？

**答案：**

1. **生产者**：`acks=0` 且网络故障
2. **Broker**：Leader 宕机，Follower 未同步
3. **消费者**：自动提交位移，消息未处理完

## 参考资料

- [Kafka 官方文档](https://kafka.apache.org/documentation/)
- [Kafka 源码分析](https://github.com/apache/kafka)
