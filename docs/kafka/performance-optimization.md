---
sidebar_position: 8
title: "性能优化"
description: "Kafka 性能调优和优化策略"
---

# Kafka 性能优化

## 性能指标

### 关键指标

| 指标         | 说明                     | 目标值              |
| ------------ | ------------------------ | ------------------- |
| **吞吐量**   | 每秒消息数/字节数        | 根据业务需求        |
| **延迟**     | 端到端延迟               | < 10ms (低延迟场景) |
| **可用性**   | 集群可用时间比例         | 99.99%              |
| **复制延迟** | Leader-Follower 同步延迟 | < 100ms             |

### 监控命令

```bash
# 查看 Topic 吞吐量
kafka-consumer-groups.sh --describe \
  --group my-group \
  --bootstrap-server localhost:9092

# 消费者延迟
kafka-consumer-groups.sh --describe \
  --group my-group \
  --bootstrap-server localhost:9092 | grep -E "LAG"
```

## 生产者优化

### 批量发送配置

```java
Properties props = new Properties();

// 批次大小（默认 16KB）
props.put("batch.size", 65536);  // 64KB

// 等待时间（默认 0ms）
props.put("linger.ms", 20);  // 等待 20ms 凑批

// 缓冲区大小
props.put("buffer.memory", 67108864);  // 64MB

// 压缩类型
props.put("compression.type", "lz4");
```

### 压缩对比

| 压缩类型   | 压缩率 | CPU 消耗 | 推荐场景         |
| ---------- | ------ | -------- | ---------------- |
| **none**   | 无     | 无       | 低延迟，CPU 敏感 |
| **gzip**   | 最高   | 高       | 带宽受限         |
| **snappy** | 中等   | 低       | 平衡选择         |
| **lz4**    | 中等   | 最低     | **生产推荐**     |
| **zstd**   | 高     | 中       | 存储优化         |

### 异步发送最佳实践

```java
// 异步发送 + 回调
for (int i = 0; i < 10000; i++) {
    ProducerRecord<String, String> record =
        new ProducerRecord<>("topic", "key-" + i, "value-" + i);

    producer.send(record, (metadata, exception) -> {
        if (exception != null) {
            // 记录失败，后续重试
            logger.error("发送失败", exception);
        }
    });

    // 每 1000 条记录 flush 一次（可选）
    if (i % 1000 == 0) {
        producer.flush();
    }
}
```

## 消费者优化

### 批量拉取配置

```java
Properties props = new Properties();

// 单次拉取最大记录数
props.put("max.poll.records", 500);

// 拉取最小字节数
props.put("fetch.min.bytes", 50000);  // 50KB

// 拉取最大等待时间
props.put("fetch.max.wait.ms", 500);  // 500ms

// 单次拉取最大字节数
props.put("fetch.max.bytes", 52428800);  // 50MB
```

### 多线程消费

```java
// 方案一：多线程处理消息
ExecutorService executor = Executors.newFixedThreadPool(10);

while (running) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));

    for (ConsumerRecord<String, String> record : records) {
        executor.submit(() -> processRecord(record));
    }

    consumer.commitAsync();
}

// 方案二：多消费者实例
int numConsumers = 3;
for (int i = 0; i < numConsumers; i++) {
    new Thread(new ConsumerRunnable(props, "topic")).start();
}
```

### 消费者积压处理

```java
// 跳过旧消息，直接消费最新
props.put("auto.offset.reset", "latest");

// 或者使用 seekToEnd
consumer.seekToEnd(consumer.assignment());
```

## Broker 优化

### 日志配置

```properties
# 日志段大小（默认 1GB）
log.segment.bytes=1073741824

# 日志保留时间
log.retention.hours=168

# 日志清理策略
log.cleanup.policy=delete

# 日志刷新策略
log.flush.interval.messages=10000
log.flush.interval.ms=1000
```

### 网络配置

```properties
# 网络线程数
num.network.threads=8

# IO 线程数
num.io.threads=16

# Socket 缓冲区
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600
```

### 副本配置

```properties
# 副本拉取线程数
num.replica.fetchers=4

# 副本拉取最大字节数
replica.fetch.max.bytes=10485760

# 副本拉取等待时间
replica.fetch.wait.max.ms=500
```

## 分区优化

### 分区数计算

```
分区数 = max(生产端吞吐量/单分区生产吞吐量, 消费端吞吐量/单分区消费吞吐量)

示例：
- 目标吞吐量: 1000 MB/s
- 单分区生产吞吐量: 100 MB/s
- 单分区消费吞吐量: 50 MB/s
- 推荐分区数 = max(10, 20) = 20
```

### 分区分配策略

```java
// 消费者分区分配策略
props.put("partition.assignment.strategy",
    "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");
```

## 操作系统优化

### 文件系统

```bash
# 使用 XFS 文件系统
mkfs.xfs /dev/sdb

# 挂载选项
mount -o noatime,nodiratime /dev/sdb /data/kafka
```

### 内核参数

```bash
# /etc/sysctl.conf

# 虚拟内存
vm.swappiness=1
vm.dirty_ratio=60
vm.dirty_background_ratio=5

# 网络
net.core.wmem_max=2097152
net.core.rmem_max=2097152
net.ipv4.tcp_wmem=4096 65536 2048000
net.ipv4.tcp_rmem=4096 65536 2048000
net.core.netdev_max_backlog=50000

# 应用配置
sysctl -p
```

### 文件描述符

```bash
# /etc/security/limits.conf
* soft nofile 100000
* hard nofile 100000
```

## JVM 优化

### G1GC 配置

```bash
export KAFKA_HEAP_OPTS="-Xms6g -Xmx6g"
export KAFKA_JVM_PERFORMANCE_OPTS="-server \
  -XX:+UseG1GC \
  -XX:MaxGCPauseMillis=20 \
  -XX:InitiatingHeapOccupancyPercent=35 \
  -XX:G1HeapRegionSize=16M \
  -XX:MinMetaspaceFreeRatio=50 \
  -XX:MaxMetaspaceFreeRatio=80"
```

### 堆内存建议

| OS 内存 | Kafka 堆内存 | 页缓存 |
| ------- | ------------ | ------ |
| 32 GB   | 6 GB         | 26 GB  |
| 64 GB   | 8 GB         | 56 GB  |
| 128 GB  | 12 GB        | 116 GB |

## 性能测试

### 生产者测试

```bash
kafka-producer-perf-test.sh \
  --topic test-topic \
  --num-records 1000000 \
  --record-size 1000 \
  --throughput -1 \
  --producer-props \
    bootstrap.servers=localhost:9092 \
    batch.size=65536 \
    linger.ms=20 \
    compression.type=lz4
```

### 消费者测试

```bash
kafka-consumer-perf-test.sh \
  --topic test-topic \
  --messages 1000000 \
  --threads 4 \
  --bootstrap-server localhost:9092
```

### 端到端延迟测试

```bash
kafka-run-class.sh kafka.tools.EndToEndLatency \
  localhost:9092 \
  test-topic \
  10000 \
  all \
  1024
```

## 性能优化清单

### 生产者

- [ ] 配置合适的 `batch.size`（32-64KB）
- [ ] 设置 `linger.ms`（10-20ms）
- [ ] 启用压缩（推荐 lz4）
- [ ] 使用异步发送 + 回调

### 消费者

- [ ] 增加 `max.poll.records`
- [ ] 配置 `fetch.min.bytes`
- [ ] 使用多线程处理
- [ ] 合理设置消费者数量

### Broker

- [ ] 增加网络和 IO 线程
- [ ] 配置日志段大小
- [ ] 调整副本拉取参数
- [ ] 使用 SSD 存储

### 操作系统

- [ ] 调整文件描述符限制
- [ ] 优化虚拟内存设置
- [ ] 配置网络参数
- [ ] 使用 XFS 文件系统

## 下一步

- 🔧 [集群管理](/docs/kafka/cluster-management) - 集群部署和管理
- 🔒 [最佳实践](/docs/kafka/best-practices) - 生产环境最佳实践
- 📊 [监控与运维](/docs/kafka/monitoring) - 监控和告警

## 参考资料

- [Kafka 性能调优指南](https://kafka.apache.org/documentation/#prodconfig)
- [LinkedIn Kafka 调优实践](https://engineering.linkedin.com/kafka)
