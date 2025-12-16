---
sidebar_position: 99
title: "消息存储机制"
description: "Kafka 的日志存储、索引、保留策略与压缩机制"
---

# Kafka 消息存储机制

Kafka 的核心设计可以概括为一句话：**把消息当成追加写的日志（Commit Log）**。

- Broker 端对每个 `Topic-Partition` 维护一条有序日志
- 生产者追加写入（append）
- 消费者通过 `offset` 顺序读取，并用索引加速定位

> 本文从“文件布局 → 索引结构 → 保留/压缩 → 副本一致性”几个维度解释 Kafka 为什么快、为什么可靠，以及你在生产环境如何做容量/保留策略设计。

## 1. Topic / Partition 到磁盘目录

Kafka 的存储粒度是 **分区（Partition）**。

典型的数据目录结构：

```
/kafka-logs/
├── orders-0/
│   ├── 00000000000000000000.log
│   ├── 00000000000000000000.index
│   ├── 00000000000000000000.timeindex
│   ├── 00000000000000000000.snapshot
│   ├── 00000000000000001024.log
│   ├── 00000000000000001024.index
│   └── 00000000000000001024.timeindex
├── orders-1/
└── ...
```

- `.log`：消息数据（Record Batch）
- `.index`：偏移量索引（Offset Index）
- `.timeindex`：时间索引（Time Index）
- `.snapshot`：用于日志压缩（Log Compaction）时的快照（不同版本略有差异）

## 2. Log Segment：分段存储

Kafka 不会把一个分区写成一个无限增长的文件，而是把日志分成多个 **Segment**（段）。

常见配置：

- `log.segment.bytes`：单个段文件的最大大小（默认 1GB）
- `log.segment.ms`：段文件滚动的最大时间（到期也会滚动）

段文件名（如 `00000000000000001024.log`）表示该 segment 的 **base offset**（该段第一条消息的 offset）。

### 为什么要分段？

- **更快的删除**：删除旧数据只要删除整段文件，不需要“挖洞”
- **更快的查找**：先定位段，再在段里通过索引定位
- **更便于压缩/清理**：对段做 compaction、截断等操作

## 3. Offset Index：偏移量索引（稀疏索引）

Kafka 的 `.index` 不是为每条消息建立索引，而是稀疏索引。

- 索引项：`<relativeOffset, position>`
  - `relativeOffset`：相对 base offset 的偏移
  - `position`：在 `.log` 文件里的物理位置

因此定位某个 offset 的流程一般是：

1. **找到目标 offset 所属的 segment**（根据 base offset）
2. 在 `.index` 中二分查找最接近但不超过目标 offset 的索引项
3. 从该 `position` 开始顺序扫描 `.log`，直到找到目标 offset

这种“**二分 + 顺序扫描**”的组合，是 Kafka 在性能和存储成本之间的经典权衡。

## 4. Time Index：按时间定位

`.timeindex` 用于根据时间戳快速定位 offset（例如“从 1 小时前开始消费”）。

典型用法：

- Consumer API：`offsetsForTimes()`
- CLI：在运维/排障时更常见（先算出时间点对应的 offset，再 `seek`）

## 5. 消息保留策略（Retention）

Kafka 的默认策略是 **删除（delete）**：消息在保留窗口内可被重复消费，超过窗口后才会被清理。

常用配置：

- `retention.ms` / `log.retention.hours`：按时间保留
- `retention.bytes`：按大小保留（每个分区）
- `log.retention.check.interval.ms`：检查过期数据的周期

> 生产上常见做法：按时间为主、按大小兜底，避免磁盘爆掉。

### 删除到底怎么发生？

Kafka 会周期性扫描 segment：

- 若一个 segment 的最大时间戳早于保留阈值，则删除该 segment
- 删除是以 segment 为单位，速度快、对在线读写影响小

## 6. Log Compaction（日志压缩）

当 `cleanup.policy=compact`（或 `compact,delete`）时，Kafka 不是简单按时间删除，而是对“同 key 的旧值”进行清理：

- 对相同 key，保留最新值
- 允许出现 Tombstone（key 的删除标记）

适用场景：

- **业务状态表**（KTable / changelog topic）
- 需要“最终状态”而不是“全量事件流”的数据

关键配置：

- `cleanup.policy=compact`
- `min.compaction.lag.ms` / `max.compaction.lag.ms`
- `delete.retention.ms`：tombstone 保留时间

> 注意：Compaction 是异步后台过程，**不保证立刻清理**。

## 7. 副本机制与一致性（Replication）

每个分区有一个 Leader 和若干 Follower。

- Producer 写入 Leader
- Follower 从 Leader 拉取数据复制
- Consumer 默认从 Leader 读（通常情况下）

### ISR（In-Sync Replicas）

ISR 是“与 Leader 保持足够同步”的副本集合。

- `acks=all` 时，Leader 需要等待 ISR 副本写入确认
- `min.insync.replicas` 控制最少需要多少 ISR 副本确认

典型组合：

- `replication.factor=3`
- `min.insync.replicas=2`
- `acks=all`

这样可以容忍 1 台 Broker 故障，同时保证高可靠。

### LEO / HW（理解延迟与一致性的关键）

- **LEO（Log End Offset）**：某副本已写入的最大 offset
- **HW（High Watermark）**：对外可见（可被消费者读取）的最大 offset

只有当消息被 ISR 副本“足够确认”后，HW 才会推进，消费者才会读到。

## 8. 为什么 Kafka 很快？（存储相关的关键点）

- **顺序写磁盘**：追加写对磁盘/SSD非常友好
- **页缓存（Page Cache）**：大量读取来自 OS page cache，而不是每次都落盘
- **零拷贝（Zero Copy）**：Broker 将磁盘数据直接发送到网卡，减少用户态/内核态拷贝
- **批处理（Record Batch）**：网络/磁盘 IO 的 amortization

## 9. 运维：查看分区磁盘占用与清理

### 查看日志目录与分区大小

```bash
kafka-log-dirs.sh --describe \
  --bootstrap-server localhost:9092
```

### 手动删除（截断）历史数据

当你需要“紧急止损”释放空间时，可以用 `kafka-delete-records.sh` 进行按 offset 截断（谨慎使用）：

```bash
# delete.json
{
  "partitions": [
    {"topic": "orders", "partition": 0, "offset": 123456}
  ],
  "version": 1
}
```

```bash
kafka-delete-records.sh \
  --bootstrap-server localhost:9092 \
  --offset-json-file delete.json
```

> 截断是不可逆的，建议先在非生产环境演练。

## 10. 下一步

- 🎯 [核心概念](/docs/kafka/core-concepts) - 分区、副本、offset 等基础概念
- ⚡ [性能优化](/docs/kafka/performance-optimization) - 批处理、压缩、参数调优
- 🔧 [集群管理](/docs/kafka/cluster-management) - 分区重分配、副本与故障处理
