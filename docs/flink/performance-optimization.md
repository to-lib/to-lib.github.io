---
sidebar_position: 16
title: "性能优化"
description: "Flink 性能优化与调优指南"
---

# Flink 性能优化

> 适用版本：Apache Flink v2.2.0

## 并行度优化

### 合理设置并行度

```java
// 全局并行度
env.setParallelism(8);

// 算子级别并行度
stream.map(...).setParallelism(16);

// Source 并行度（通常等于 Kafka 分区数）
kafkaSource.setParallelism(partitionCount);
```

### 并行度设置原则

- **Source**：等于数据源分区数
- **计算算子**：根据 CPU 核数和任务槽数
- **Sink**：根据下游系统承载能力
- **避免过度并行**：过多并行度增加协调开销

## 内存优化

### 托管内存配置

```yaml
# 增加托管内存用于 RocksDB
taskmanager.memory.managed.fraction: 0.4

# 网络缓冲区
taskmanager.memory.network.fraction: 0.1
taskmanager.memory.network.min: 64mb
taskmanager.memory.network.max: 1gb
```

### 对象重用

```java
// 启用对象重用（减少 GC）
env.getConfig().enableObjectReuse();
```

> ⚠️ **注意**：启用对象重用后，不要修改发出的对象或将其存储在状态中。

### 序列化优化

```java
// 使用高效的 POJO 序列化
public class Event {
    public String id;
    public long timestamp;
    public double value;
    // 必须有无参构造函数
    public Event() {}
}

// 注册类型（提高序列化效率）
env.registerType(Event.class);

// 使用 Kryo 自定义序列化
env.getConfig().registerTypeWithKryoSerializer(
    MyClass.class,
    MySerializer.class
);
```

## 状态优化

### 选择合适的状态后端

```java
// 小状态：HashMapStateBackend
env.setStateBackend(new HashMapStateBackend());

// 大状态：RocksDB + 增量检查点
EmbeddedRocksDBStateBackend rocksdb = new EmbeddedRocksDBStateBackend(true);
env.setStateBackend(rocksdb);
```

### RocksDB 调优

```yaml
# 增加写缓冲区数量
state.backend.rocksdb.writebuffer.count: 4

# 增加写缓冲区大小
state.backend.rocksdb.writebuffer.size: 64mb

# 增加块缓存
state.backend.rocksdb.block.cache-size: 256mb

# 启用预定义选项
state.backend.rocksdb.predefined-options: SPINNING_DISK_OPTIMIZED_HIGH_MEM
```

### 状态 TTL

```java
StateTtlConfig ttlConfig = StateTtlConfig
    .newBuilder(Time.hours(24))
    .setUpdateType(UpdateType.OnCreateAndWrite)
    .cleanupIncrementally(10, true)  // 增量清理
    .build();
```

## 检查点优化

### 检查点配置

```java
// 检查点间隔
env.enableCheckpointing(60000);

// 最小间隔
env.getCheckpointConfig().setMinPauseBetweenCheckpoints(30000);

// 超时时间
env.getCheckpointConfig().setCheckpointTimeout(600000);

// 并发检查点数
env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);

// 非对齐检查点（减少背压影响）
env.getCheckpointConfig().enableUnalignedCheckpoints();
```

### 增量检查点

```java
EmbeddedRocksDBStateBackend rocksdb =
    new EmbeddedRocksDBStateBackend(true); // 启用增量检查点
env.setStateBackend(rocksdb);
```

## 网络优化

### 缓冲区配置

```yaml
# 每个通道的缓冲区数量
taskmanager.network.memory.buffers-per-channel: 2

# 浮动缓冲区数量
taskmanager.network.memory.floating-buffers-per-gate: 8

# 网络超时
taskmanager.network.request-backoff.max: 10000
```

### 批量发送

```yaml
# 输出缓冲区刷新间隔
execution.buffer-timeout: 100ms
```

## 数据倾斜处理

### 预聚合 + 随机前缀

```java
// 第一阶段：添加随机前缀，局部聚合
stream
    .map(e -> new Tuple2<>(e.getKey() + "_" + random.nextInt(10), e.getValue()))
    .keyBy(t -> t.f0)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .reduce((a, b) -> new Tuple2<>(a.f0, a.f1 + b.f1))
    // 第二阶段：去除前缀，全局聚合
    .map(t -> new Tuple2<>(t.f0.split("_")[0], t.f1))
    .keyBy(t -> t.f0)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .reduce((a, b) -> new Tuple2<>(a.f0, a.f1 + b.f1));
```

### 使用 rebalance

```java
// 强制重新分区
stream.rebalance().map(...);
```

## 窗口优化

### 增量聚合

```java
// 使用 ReduceFunction（增量计算）
stream
    .keyBy(...)
    .window(...)
    .reduce(new MyReduceFunction());

// 使用 AggregateFunction（更灵活的增量计算）
stream
    .keyBy(...)
    .window(...)
    .aggregate(new MyAggregateFunction());
```

### 避免全量计算

```java
// ❌ 不推荐：ProcessWindowFunction 会缓存所有元素
.process(new ProcessWindowFunction<...>() {...})

// ✅ 推荐：结合 reduce 和 process
.reduce(reduceFunction, processFunction)
```

## Source 优化

### Kafka Source 配置

```java
KafkaSource.<String>builder()
    .setProperty("fetch.min.bytes", "1048576")  // 批量拉取
    .setProperty("fetch.max.wait.ms", "500")
    .setProperty("max.poll.records", "10000")
    .build();
```

## 监控与诊断

### 关键指标

```
# 背压检测
flink_taskmanager_job_task_isBackPressured

# 检查点耗时
flink_jobmanager_job_lastCheckpointDuration

# 水印延迟
flink_taskmanager_job_task_operator_currentInputWatermark

# 记录延迟
flink_taskmanager_job_task_operator_recordsLagMax
```

### 性能诊断

1. **检查背压**：Web UI → Job → Task → BackPressure
2. **分析火焰图**：Thread Dump 分析
3. **监控 GC**：GC 日志分析

## 优化清单

| 优化项       | 配置                  | 效果             |
| ------------ | --------------------- | ---------------- |
| 增量检查点   | RocksDB + incremental | 减少检查点大小   |
| 非对齐检查点 | unaligned checkpoints | 减少背压延迟     |
| 对象重用     | enableObjectReuse()   | 减少 GC          |
| 本地聚合     | reduce/aggregate      | 减少网络传输     |
| 异步 IO      | AsyncDataStream       | 提高外部调用效率 |

## 下一步

- 📋 [最佳实践](/docs/flink/best-practices) - 开发规范
- 🔧 [部署与运维](/docs/flink/deployment) - 生产部署
- ❓ [常见问题](/docs/flink/faq) - FAQ
