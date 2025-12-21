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

- 消息队列：替代传统 MQ，提供更高吞吐量
- 日志聚合：收集分布式日志
- 流处理：实时数据处理和分析
- 事件溯源：记录状态变更
- 指标监控：收集应用指标
- 数据管道：连接不同数据系统

### 2. Kafka 的核心组件有哪些？

**答案：**

| 组件                | 说明                      |
| ------------------- | ------------------------- |
| **Producer**        | 生产者，发送消息到 Topic  |
| **Consumer**        | 消费者，从 Topic 消费消息 |
| **Broker**          | Kafka 服务器节点          |
| **Topic**           | 消息分类，逻辑概念        |
| **Partition**       | 分区，Topic 的物理划分    |
| **Replica**         | 副本，保证高可用          |
| **ZooKeeper/KRaft** | 集群协调与元数据管理      |

### 3. Kafka 中的 Topic 和 Partition 是什么关系？

**答案：**

- **Topic** 是逻辑概念，代表一类消息
- **Partition** 是物理概念，是 Topic 的分片
- 一个 Topic 可以有多个 Partition
- Partition 实现了并行处理和水平扩展
- 每个 Partition 内消息有序，跨 Partition 无序

```
Topic "orders" → Partition 0 → [msg1, msg2, msg3...]
              → Partition 1 → [msg4, msg5, msg6...]
              → Partition 2 → [msg7, msg8, msg9...]
```

## 架构与原理

### 4. Kafka 如何保证消息不丢失？

**答案：**
需要三方面配合：

**生产者端：**

```java
props.put("acks", "all");                    // 等待所有 ISR 确认
props.put("retries", Integer.MAX_VALUE);     // 重试
props.put("enable.idempotence", "true");     // 幂等性
```

**Broker 端：**

```properties
replication.factor >= 3
min.insync.replicas >= 2
unclean.leader.election.enable = false
```

**消费者端：**

```java
props.put("enable.auto.commit", "false");    // 关闭自动提交
consumer.commitSync();                        // 处理完后手动提交
```

### 5. Kafka 如何保证消息的顺序性？

**答案：**

- **分区内有序**：同一分区内消息严格按发送顺序存储和消费
- **全局有序**：Topic 只设置 1 个分区（牺牲并行性）
- **业务有序**：相同 key 的消息发送到同一分区

```java
// 使用 orderId 作为 key，保证同一订单的消息有序
producer.send(new ProducerRecord<>("orders", orderId, message));
```

**注意事项：**

- 开启重试可能导致乱序，需设置 `max.in.flight.requests.per.connection=1`
- 或使用幂等性生产者（`enable.idempotence=true`）

### 6. 什么是 ISR？它的作用是什么？

**答案：**

**ISR**（In-Sync Replicas）是同步副本集合，包含：

- Leader 副本
- 与 Leader 保持同步的 Follower 副本

**作用：**

1. 只有 ISR 中的副本才有资格被选举为 Leader
2. `acks=all` 时，只需等待 ISR 中的副本确认
3. 通过 `min.insync.replicas` 控制最小同步副本数

**副本落后原因：**

- 网络延迟
- Follower 处理速度慢
- GC 停顿

### 7. 什么是 Controller？它有什么作用？

**答案：**

Controller 是集群中的特殊 Broker，负责：

1. **分区 Leader 选举**：Broker 宕机时选举新 Leader
2. **Topic 管理**：处理 Topic 创建、删除
3. **分区重分配**：调整分区到不同 Broker
4. **副本状态同步**：维护集群元数据

Controller 选举机制：

- 传统模式：通过 ZooKeeper 选举
- KRaft 模式：通过 Raft 协议选举

### 8. 消费者组是如何工作的？

**答案：**

消费者组（Consumer Group）特点：

1. **负载均衡**：一个分区只能被组内一个消费者消费
2. **故障转移**：消费者宕机后，分区自动分配给其他消费者
3. **独立消费**：不同消费者组可以独立消费相同 Topic

**分区分配策略：**

| 策略                        | 说明                      |
| --------------------------- | ------------------------- |
| `RangeAssignor`             | 按 Topic 范围分配（默认） |
| `RoundRobinAssignor`        | 轮询分配                  |
| `StickyAssignor`            | 粘性分配，减少再均衡      |
| `CooperativeStickyAssignor` | 协作粘性，增量再均衡      |

## 性能优化

### 9. Kafka 为什么这么快？

**答案：**

1. **顺序写磁盘**：追加写入，比随机写内存还快
2. **零拷贝**：使用 `sendfile` 系统调用，减少数据拷贝
3. **批量处理**：批量发送和消费，减少网络开销
4. **分区并行**：多分区并行读写
5. **页缓存**：利用 OS 页缓存，避免 JVM GC
6. **压缩**：减少网络传输和磁盘存储

零拷贝原理图：

```
传统方式：磁盘 → 内核缓冲区 → 用户缓冲区 → socket缓冲区 → 网卡
零拷贝：  磁盘 → 内核缓冲区 → 网卡
```

### 10. 如何优化 Kafka 生产者性能？

**答案：**

```java
// 1. 增大批次大小
props.put("batch.size", 32768);

// 2. 增加等待时间，让更多消息累积
props.put("linger.ms", 20);

// 3. 启用压缩
props.put("compression.type", "lz4");

// 4. 增大缓冲区
props.put("buffer.memory", 67108864);

// 5. 使用异步发送
producer.send(record, (metadata, exception) -> {
    if (exception != null) {
        log.error("Send failed", exception);
    }
});
```

### 11. 如何优化 Kafka 消费者性能？

**答案：**

1. **增加消费者数量**（不超过分区数）
2. **调整拉取参数**：
   ```java
   props.put("fetch.min.bytes", 1024);        // 最小拉取字节
   props.put("fetch.max.wait.ms", 500);       // 最大等待时间
   props.put("max.poll.records", 500);        // 单次拉取数量
   ```
3. **多线程处理**：拉取和处理分离
4. **批量处理**：累积一批消息后批量入库
5. **异步提交**：使用 `commitAsync()` 减少阻塞

## 高级特性

### 12. Kafka 事务是如何实现的？

**答案：**

Kafka 事务保证跨分区的原子写入：

```java
// 1. 配置事务 ID
props.put("transactional.id", "my-transactional-id");

// 2. 初始化事务
producer.initTransactions();

try {
    // 3. 开始事务
    producer.beginTransaction();

    // 4. 发送消息
    producer.send(new ProducerRecord<>("topic1", "msg1"));
    producer.send(new ProducerRecord<>("topic2", "msg2"));

    // 5. 提交事务
    producer.commitTransaction();
} catch (Exception e) {
    // 6. 回滚事务
    producer.abortTransaction();
}
```

事务实现机制：

- **Transaction Coordinator**：管理事务状态
- **两阶段提交**：prepare → commit
- **事务日志**：`__transaction_state` 内部 Topic

### 13. 什么是 Exactly-Once 语义？如何实现？

**答案：**

精确一次语义保证消息既不丢失也不重复：

**实现方式：**

1. **幂等生产者**：

   ```java
   props.put("enable.idempotence", "true");
   ```

   - 每个生产者分配唯一 PID
   - 每条消息分配序列号
   - Broker 去重

2. **事务 + 幂等消费**：

   ```java
   // 消费者设置
   props.put("isolation.level", "read_committed");
   ```

3. **Kafka Streams**：
   ```java
   props.put("processing.guarantee", "exactly_once_v2");
   ```

### 14. 什么是消费者再均衡？如何避免频繁再均衡？

**答案：**

**再均衡**（Rebalance）：消费者组成员变化时，重新分配分区。

**触发条件：**

- 消费者加入或离开
- 消费者心跳超时
- 订阅 Topic 变化

**避免频繁再均衡：**

```java
// 1. 增大超时时间
props.put("session.timeout.ms", 45000);
props.put("heartbeat.interval.ms", 15000);
props.put("max.poll.interval.ms", 600000);

// 2. 减少处理时间
props.put("max.poll.records", 100);

// 3. 使用静态成员（Kafka 2.3+）
props.put("group.instance.id", "consumer-1");
```

### 15. Kafka 的日志存储结构是怎样的？

**答案：**

```
/kafka-logs/
├── my-topic-0/                    # Topic 分区目录
│   ├── 00000000000000000000.log   # 日志分段
│   ├── 00000000000000000000.index # 偏移量索引
│   ├── 00000000000000000000.timeindex # 时间索引
│   ├── 00000000000012345678.log   # 新分段
│   └── ...
├── my-topic-1/
└── ...
```

**关键概念：**

- **Log Segment**：日志分段，默认 1GB
- **Index**：稀疏索引，加速消息查找
- **TimeIndex**：时间索引，支持按时间查找

### 16. Kafka 如何根据 offset 快速定位消息？

**答案：**

**二分查找 + 稀疏索引：**

1. 根据 offset 二分查找对应的 segment 文件
2. 在 `.index` 文件中二分查找，定位物理位置
3. 在 `.log` 文件中顺序扫描找到目标消息

```
Index 文件格式：
offset: 100 → position: 0
offset: 200 → position: 4096
offset: 300 → position: 8192
```

### 17. 什么是零拷贝？Kafka 是如何使用的？

**答案：**

**零拷贝**：避免数据在内核空间和用户空间之间复制。

**传统方式（4 次拷贝）：**

```
磁盘 → 内核缓冲区 → 用户缓冲区 → socket缓冲区 → 网卡
```

**零拷贝（2 次拷贝）：**

```
磁盘 → 内核缓冲区 → 网卡（sendfile 系统调用）
```

**Kafka 使用场景：**

- 消费者拉取消息
- Follower 从 Leader 同步数据

## 运维相关

### 18. Kafka 如何进行扩容？

**答案：**

**1. 添加 Broker：**

- 配置新 Broker 并启动
- 新 Broker 自动加入集群

**2. 分区重分配：**

```bash
# 生成迁移计划
kafka-reassign-partitions.sh --generate \
  --bootstrap-server localhost:9092 \
  --topics-to-move-json-file topics.json \
  --broker-list "1,2,3,4"

# 执行迁移
kafka-reassign-partitions.sh --execute \
  --bootstrap-server localhost:9092 \
  --reassignment-json-file plan.json

# 验证完成
kafka-reassign-partitions.sh --verify \
  --bootstrap-server localhost:9092 \
  --reassignment-json-file plan.json
```

### 19. Kafka 丢消息的场景有哪些？

**答案：**

| 位置       | 场景                  | 解决方案                               |
| ---------- | --------------------- | -------------------------------------- |
| **生产者** | `acks=0`，网络故障    | 设置 `acks=all`                        |
| **生产者** | 缓冲区满，消息丢弃    | 增大 `buffer.memory`                   |
| **Broker** | Leader 宕机，ISR 落后 | 增加 `min.insync.replicas`             |
| **Broker** | 脏选举                | `unclean.leader.election.enable=false` |
| **消费者** | 自动提交，处理失败    | 手动提交 offset                        |
| **消费者** | 先提交后处理          | 先处理后提交                           |

### 20. 如何处理消费者 Lag 过高？

**答案：**

**排查步骤：**

```bash
# 查看消费者 Lag
kafka-consumer-groups.sh --describe \
  --group my-group \
  --bootstrap-server localhost:9092
```

**解决方案：**

1. **增加消费者实例**
2. **增加分区数**（需评估影响）
3. **优化消费者处理逻辑**
4. **跳过积压消息**（业务允许时）：
   ```bash
   kafka-consumer-groups.sh --reset-offsets --to-latest \
     --group my-group --topic my-topic \
     --bootstrap-server localhost:9092 --execute
   ```

### 21. Kafka 和 RocketMQ 有什么区别？

**答案：**

| 特性         | Kafka            | RocketMQ       |
| ------------ | ---------------- | -------------- |
| **设计目标** | 日志收集、流处理 | 业务消息       |
| **延迟消息** | 不支持原生       | 支持任意延迟   |
| **事务消息** | 支持             | 支持（更成熟） |
| **消息过滤** | 不支持           | 支持 Tag/SQL   |
| **吞吐量**   | 更高             | 高             |
| **消息回溯** | 完整支持         | 支持           |
| **社区生态** | 更丰富           | 阿里主导       |

### 22. KRaft 模式和 ZooKeeper 模式有什么区别？

**答案：**

| 特性           | ZooKeeper 模式 | KRaft 模式                  |
| -------------- | -------------- | --------------------------- |
| **元数据存储** | ZooKeeper      | Kafka 内部日志              |
| **运维复杂度** | 需维护两套系统 | 只需维护 Kafka              |
| **扩展性**     | 受 ZK 限制     | 更好的扩展性                |
| **可用性**     | Kafka 3.0+     | Kafka 3.0+（生产可用 3.3+） |
| **性能**       | 受 ZK 延迟影响 | 更低延迟                    |

**启动 KRaft 模式：**

```bash
# 生成集群 ID
KAFKA_CLUSTER_ID="$(bin/kafka-storage.sh random-uuid)"

# 格式化存储
bin/kafka-storage.sh format -t $KAFKA_CLUSTER_ID -c config/kraft/server.properties

# 启动
bin/kafka-server-start.sh config/kraft/server.properties
```

## 参考资料

- [Kafka 官方文档](https://kafka.apache.org/documentation/)
- [Kafka 源码分析](https://github.com/apache/kafka)
- [Confluent 博客](https://www.confluent.io/blog/)
