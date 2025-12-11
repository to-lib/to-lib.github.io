---
sidebar_position: 10
title: "状态管理"
description: "Flink 状态管理与容错机制详解"
---

# Flink 状态管理

## 概述

状态管理是 Flink 的核心能力之一，使得 Flink 能够处理复杂的有状态计算，如聚合、窗口、机器学习模型等。

## 状态类型

### Keyed State

基于 KeyedStream 的状态，每个 Key 维护独立的状态：

```java
public class CountFunction extends RichFlatMapFunction<Event, Result> {
    // 声明状态
    private ValueState<Long> countState;

    @Override
    public void open(Configuration parameters) {
        // 创建状态描述符
        ValueStateDescriptor<Long> descriptor =
            new ValueStateDescriptor<>("count", Long.class);
        // 获取状态
        countState = getRuntimeContext().getState(descriptor);
    }

    @Override
    public void flatMap(Event event, Collector<Result> out) throws Exception {
        Long count = countState.value();
        count = count == null ? 1L : count + 1;
        countState.update(count);
        out.collect(new Result(event.getKey(), count));
    }
}
```

### Keyed State 类型

| 状态类型                        | 描述                          | 适用场景           |
| ------------------------------- | ----------------------------- | ------------------ |
| **ValueState\<T\>**             | 单个值                        | 计数器、累加器     |
| **ListState\<T\>**              | 元素列表                      | 事件缓存、历史记录 |
| **MapState\<K, V\>**            | 键值对映射                    | 索引、查找表       |
| **ReducingState\<T\>**          | 聚合值（需要 ReduceFunction） | 求和、求最大值     |
| **AggregatingState\<IN, OUT\>** | 复杂聚合                      | 平均值、自定义聚合 |

### ListState 示例

```java
private ListState<Event> eventBuffer;

@Override
public void open(Configuration parameters) {
    ListStateDescriptor<Event> descriptor =
        new ListStateDescriptor<>("events", Event.class);
    eventBuffer = getRuntimeContext().getListState(descriptor);
}

@Override
public void processElement(Event event, Context ctx, Collector<Result> out)
        throws Exception {
    eventBuffer.add(event);

    // 批量处理
    if (shouldFlush()) {
        for (Event e : eventBuffer.get()) {
            out.collect(process(e));
        }
        eventBuffer.clear();
    }
}
```

### MapState 示例

```java
private MapState<String, Integer> wordCounts;

@Override
public void open(Configuration parameters) {
    MapStateDescriptor<String, Integer> descriptor =
        new MapStateDescriptor<>("wordCounts", String.class, Integer.class);
    wordCounts = getRuntimeContext().getMapState(descriptor);
}

@Override
public void processElement(String word, Context ctx, Collector<Result> out)
        throws Exception {
    Integer count = wordCounts.get(word);
    count = count == null ? 1 : count + 1;
    wordCounts.put(word, count);
    out.collect(new Result(word, count));
}
```

### Operator State

算子级别的状态，不按 Key 分区：

```java
public class BufferingSink implements SinkFunction<Event>,
        CheckpointedFunction {

    private List<Event> bufferedElements;
    private ListState<Event> checkpointedState;

    @Override
    public void snapshotState(FunctionSnapshotContext context) throws Exception {
        checkpointedState.clear();
        for (Event element : bufferedElements) {
            checkpointedState.add(element);
        }
    }

    @Override
    public void initializeState(FunctionInitializationContext context)
            throws Exception {
        ListStateDescriptor<Event> descriptor =
            new ListStateDescriptor<>("buffered-elements", Event.class);

        checkpointedState = context.getOperatorStateStore()
            .getListState(descriptor);

        if (context.isRestored()) {
            for (Event element : checkpointedState.get()) {
                bufferedElements.add(element);
            }
        }
    }
}
```

## 状态后端

### HashMapStateBackend

将状态保存在 TaskManager 的 JVM 堆内存中：

```java
env.setStateBackend(new HashMapStateBackend());
env.getCheckpointConfig().setCheckpointStorage("file:///checkpoints");
```

**特点**：

- ✅ 访问速度快
- ❌ 状态大小受限于内存
- 适用于：小状态、开发测试

### EmbeddedRocksDBStateBackend

将状态保存在 RocksDB 中：

```java
env.setStateBackend(new EmbeddedRocksDBStateBackend());
env.getCheckpointConfig().setCheckpointStorage("hdfs:///checkpoints");
```

**特点**：

- ✅ 支持超大状态（TB 级）
- ✅ 支持增量检查点
- ❌ 访问速度较慢
- 适用于：大状态、生产环境

### RocksDB 配置优化

```java
EmbeddedRocksDBStateBackend rocksdb = new EmbeddedRocksDBStateBackend();
rocksdb.setDbStoragePath("/data/rocksdb");
rocksdb.setPredefinedOptions(PredefinedOptions.SPINNING_DISK_OPTIMIZED);
env.setStateBackend(rocksdb);
```

## 检查点（Checkpoint）

### 配置检查点

```java
// 启用检查点，间隔 5 分钟
env.enableCheckpointing(300000);

// 精确一次语义
env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);

// 检查点超时时间
env.getCheckpointConfig().setCheckpointTimeout(600000);

// 检查点之间最小间隔
env.getCheckpointConfig().setMinPauseBetweenCheckpoints(60000);

// 最大同时进行的检查点数量
env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);

// 作业取消时保留检查点
env.getCheckpointConfig().setExternalizedCheckpointCleanup(
    ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
```

### 检查点存储

```java
// 文件系统
env.getCheckpointConfig().setCheckpointStorage("file:///checkpoints");

// HDFS
env.getCheckpointConfig().setCheckpointStorage("hdfs:///flink/checkpoints");

// S3
env.getCheckpointConfig().setCheckpointStorage("s3://bucket/checkpoints");
```

## 保存点（Savepoint）

### 触发保存点

```bash
# 触发保存点
flink savepoint <jobId> hdfs:///savepoints

# 取消作业并创建保存点
flink cancel -s hdfs:///savepoints <jobId>
```

### 从保存点恢复

```bash
flink run -s hdfs:///savepoints/savepoint-xxx myJob.jar
```

## 状态 TTL

设置状态过期策略：

```java
StateTtlConfig ttlConfig = StateTtlConfig
    .newBuilder(Time.days(7))  // 7 天过期
    .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
    .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
    .cleanupFullSnapshot()  // 检查点时清理
    .build();

ValueStateDescriptor<String> stateDescriptor =
    new ValueStateDescriptor<>("myState", String.class);
stateDescriptor.enableTimeToLive(ttlConfig);
```

### TTL 清理策略

```java
// 增量清理（每访问 N 条记录清理一次）
.cleanupIncrementally(10, true)

// RocksDB 压缩时清理
.cleanupInRocksdbCompactFilter(1000)

// 检查点时全量清理
.cleanupFullSnapshot()
```

## Broadcast State

将小数据集广播到所有并行任务：

```java
// 定义广播状态描述符
MapStateDescriptor<String, Rule> ruleStateDescriptor =
    new MapStateDescriptor<>("rules", String.class, Rule.class);

// 创建广播流
BroadcastStream<Rule> ruleBroadcastStream =
    ruleStream.broadcast(ruleStateDescriptor);

// 连接数据流和广播流
dataStream
    .connect(ruleBroadcastStream)
    .process(new BroadcastProcessFunction<Event, Rule, Result>() {
        @Override
        public void processElement(Event event, ReadOnlyContext ctx,
                Collector<Result> out) {
            // 读取广播状态
            ReadOnlyBroadcastState<String, Rule> state =
                ctx.getBroadcastState(ruleStateDescriptor);
            Rule rule = state.get(event.getRuleId());
            // 应用规则处理事件
        }

        @Override
        public void processBroadcastElement(Rule rule, Context ctx,
                Collector<Result> out) {
            // 更新广播状态
            ctx.getBroadcastState(ruleStateDescriptor).put(rule.getId(), rule);
        }
    });
```

## 下一步

- 📊 [Table API & SQL](./table-sql.md) - 声明式处理
- ⚡ [CEP 复杂事件处理](./cep.md) - 模式匹配
- 🚀 [性能优化](./performance-optimization.md) - 调优指南
