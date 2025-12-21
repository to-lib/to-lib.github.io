---
sidebar_position: 9
title: Flink 面试题
description: "Flink 面试常见问题及答案"
---

# Flink 面试题精选

> 适用版本：Apache Flink v2.2.0

## 基础概念

### 1. 什么是 Apache Flink？

Apache Flink 是一个开源的分布式流处理框架，具有以下特点：

- 真正的流处理（不是微批）
- 内置状态管理
- 精确一次语义
- 毫秒级延迟
- 统一的批流处理 API

### 2. Flink 的核心组件有哪些？

- **JobManager**：作业管理器，负责调度和协调
- **TaskManager**：任务管理器，执行具体任务
- **Dispatcher**：接收作业提交，启动 JobManager
- **ResourceManager**：资源管理，分配 TaskManager 槽位

### 3. 什么是 Operator Chain？

算子链是 Flink 的一种优化机制，将多个算子合并在同一个线程中执行，减少序列化和网络开销。

**禁用条件**：

- 不同并行度
- 不同的共享组
- 显式禁用

## 时间与窗口

### 4. Flink 支持哪些时间语义？

| 时间类型        | 描述         | 使用场景     |
| --------------- | ------------ | ------------ |
| Event Time      | 事件发生时间 | 需要精确结果 |
| Processing Time | 处理时间     | 对延迟敏感   |
| Ingestion Time  | 摄入时间     | 折中方案     |

### 5. 什么是 Watermark？如何处理迟到数据？

**Watermark** 是处理乱序数据的机制，表示该时间点之前的数据应该都已到达。

**处理迟到数据的方式**：

1. `allowedLateness()`：允许窗口延迟关闭
2. `sideOutputLateData()`：侧输出延迟数据
3. 增大 Watermark 延迟

### 6. Flink 窗口类型有哪些？

- **滚动窗口**：固定大小，不重叠
- **滑动窗口**：固定大小，可重叠
- **会话窗口**：基于活动间隙
- **全局窗口**：需自定义触发器

## 状态管理

### 7. Flink 有哪些状态类型？

**Keyed State（键控状态）**：

- ValueState
- ListState
- MapState
- ReducingState
- AggregatingState

**Operator State（算子状态）**：

- ListState
- UnionListState
- BroadcastState

### 8. Checkpoint 和 Savepoint 有什么区别？

| 特征     | Checkpoint       | Savepoint  |
| -------- | ---------------- | ---------- |
| 触发方式 | 自动             | 手动       |
| 主要用途 | 故障恢复         | 升级/迁移  |
| 格式     | 内部格式         | 可移植格式 |
| 生命周期 | 作业完成后可删除 | 用户管理   |

### 9. 如何保证 Exactly-Once 语义？

1. **检查点机制**：定期快照状态
2. **两阶段提交**：与外部系统配合
3. **幂等写入**：Sink 端去重

## 性能优化

### 10. 如何解决数据倾斜问题？

1. **预聚合**：先局部聚合再全局聚合
2. **随机前缀**：打散热点 Key
3. **rebalance()**：强制重新分区
4. **优化数据分布**：调整业务逻辑

### 11. Flink 有哪些调优手段？

- **资源调优**：合理设置并行度和内存
- **算子调优**：启用对象重用、优化序列化
- **网络调优**：调整缓冲区大小
- **状态调优**：选择合适的状态后端

## 实战问题

### 12. 如何实现双流 Join？

```java
// 窗口 Join
stream1.join(stream2)
    .where(e -> e.getKey())
    .equalTo(e -> e.getKey())
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .apply((e1, e2) -> new Result(e1, e2));

// Interval Join
stream1.keyBy(e -> e.getKey())
    .intervalJoin(stream2.keyBy(e -> e.getKey()))
    .between(Time.minutes(-5), Time.minutes(5))
    .process(new ProcessJoinFunction<>() { ... });
```

### 13. 如何实现 TopN 查询？

使用 Flink SQL：

```sql
SELECT * FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) AS rn
  FROM products
) WHERE rn <= 10;
```

### 14. 生产环境选择什么部署模式？

推荐 **Application Mode**：

- main() 在集群执行
- 每个作业独立隔离
- 启动速度快
- 资源利用率高

## 高级主题

### 15. Flink CDC 是什么？

Flink CDC 是一个基于变更数据捕获的数据集成框架，可以实时同步数据库变更到 Flink：

```sql
CREATE TABLE mysql_source (
  id INT,
  name STRING,
  PRIMARY KEY (id) NOT ENFORCED
) WITH (
  'connector' = 'mysql-cdc',
  'hostname' = 'localhost',
  'database-name' = 'mydb',
  'table-name' = 'users'
);
```

### 16. 如何处理 Flink 作业的反压？

**诊断反压**：

- 通过 Web UI 查看 Backpressure 指标
- 检查 `isBackPressured` 监控值

**解决方案**：

1. 增加算子并行度
2. 优化算子逻辑（减少耗时操作）
3. 调整网络缓冲区配置
4. 检查下游系统性能

### 17. Flink SQL 和 DataStream API 如何选择？

| 场景         | 推荐 API       |
| ------------ | -------------- |
| 简单 ETL     | Flink SQL      |
| 复杂业务逻辑 | DataStream API |
| 快速原型开发 | Flink SQL      |
| 需要精细控制 | DataStream API |
| 多表关联     | Flink SQL      |
| CEP 模式匹配 | DataStream API |

### 18. RocksDB 状态后端的优缺点？

**优点**：

- 支持超大状态（TB 级）
- 支持增量检查点
- 堆外存储，不影响 GC

**缺点**：

- 访问速度比内存慢
- 需要序列化/反序列化
- 配置相对复杂

```java
env.setStateBackend(new EmbeddedRocksDBStateBackend(true)); // 启用增量检查点
```

### 19. 如何保证 Flink 作业的幂等性？

1. **Sink 端去重**：使用唯一键更新
2. **Two-Phase Commit**：配合支持事务的 Sink
3. **Upsert 模式**：使用 upsert-kafka 等连接器
4. **业务幂等设计**：使用唯一业务 ID

### 20. Flink 集群的部署模式有哪些？

| 模式                 | 特点     | 适用场景     |
| -------------------- | -------- | ------------ |
| **Standalone**       | 简单部署 | 开发测试     |
| **YARN Session**     | 共享资源 | 多作业环境   |
| **YARN Per-Job**     | 资源隔离 | 生产环境     |
| **Kubernetes**       | 云原生   | K8s 环境     |
| **Application Mode** | 推荐模式 | 生产最佳实践 |

### 21. 如何监控 Flink 作业？

1. **Web UI**：内置监控界面
2. **Metrics**：集成 Prometheus + Grafana
3. **日志**：配置日志聚合
4. **告警**：配置 AlertManager 规则

**关键监控指标**：

- 吞吐量：`numRecordsIn/Out`
- 延迟：`latency` 指标
- 检查点：持续时间和失败次数
- 背压：`isBackPressured`

### 22. Flink 如何实现端到端精确一次？

1. **Source**：可重放（如 Kafka offset）
2. **State**：检查点保证状态一致
3. **Sink**：两阶段提交或幂等写入

```java
// Kafka 端到端精确一次
KafkaSink<String> sink = KafkaSink.<String>builder()
    .setDeliveryGuarantee(DeliveryGuarantee.EXACTLY_ONCE)
    .setTransactionalIdPrefix("flink-")
    .build();
```

### 23. 什么是 Flink 的 Unaligned Checkpoint？

非对齐检查点允许在处理 Barrier 时继续处理数据，避免背压传播。

**优点**：

- 减少检查点时间
- 避免背压时检查点超时

**缺点**：

- 状态恢复时间可能增加
- 不支持所有场景

```java
env.getCheckpointConfig().enableUnalignedCheckpoints();
```

### 24. 如何优化 Flink SQL 作业性能？

1. **使用 MiniBatch**：

```sql
SET 'table.exec.mini-batch.enabled' = 'true';
SET 'table.exec.mini-batch.size' = '5000';
```

2. **本地聚合**：

```sql
SET 'table.optimizer.agg-phase-strategy' = 'TWO_PHASE';
```

3. **优化 Join**：

- 使用 Lookup Join 代替 Regular Join
- 配置维表缓存

### 25. Flink 状态过期如何配置？

```java
StateTtlConfig ttlConfig = StateTtlConfig
    .newBuilder(Time.days(7))
    .setUpdateType(UpdateType.OnCreateAndWrite)
    .setStateVisibility(StateVisibility.NeverReturnExpired)
    .cleanupIncrementally(10, true)  // 增量清理
    .build();

stateDescriptor.enableTimeToLive(ttlConfig);
```

**清理策略**：

- `cleanupFullSnapshot()`：检查点时清理
- `cleanupIncrementally()`：访问时增量清理
- `cleanupInRocksdbCompactFilter()`：RocksDB 压缩时清理
