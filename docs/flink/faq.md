---
sidebar_position: 8
title: "常见问题"
description: "Flink 常见问题解答"
---

# Flink 常见问题

## 基础概念

### Flink 和 Spark Streaming 有什么区别？

| 对比项   | Flink            | Spark Streaming   |
| -------- | ---------------- | ----------------- |
| 处理模型 | 真正的流处理     | 微批处理          |
| 延迟     | 毫秒级           | 秒级              |
| 状态管理 | 原生支持         | 需要额外组件      |
| 精确一次 | 原生支持         | 需配置            |
| API      | DataStream/Table | DStream/DataFrame |

### 什么是 Event Time 和 Processing Time？

- **Event Time**：事件实际发生的时间，嵌入在数据中
- **Processing Time**：Flink 系统处理事件的时间

推荐使用 Event Time，因为它能处理乱序和延迟数据。

### 什么是水印（Watermark）？

水印是 Flink 用来处理乱序数据的机制。它表示"不会再有比这个时间更早的数据到来"。

```java
WatermarkStrategy.<Event>forBoundedOutOfOrderness(Duration.ofSeconds(5))
```

## 状态与检查点

### 检查点和保存点有什么区别？

| 特性     | 检查点   | 保存点         |
| -------- | -------- | -------------- |
| 触发方式 | 自动     | 手动           |
| 用途     | 故障恢复 | 版本升级、迁移 |
| 格式     | 内部格式 | 标准格式       |
| 存储     | 可配置   | 外部存储       |

### 如何处理状态过大的问题？

1. 使用 RocksDB 状态后端
2. 配置增量检查点
3. 设置状态 TTL
4. 优化数据结构

```java
StateTtlConfig ttlConfig = StateTtlConfig
    .newBuilder(Time.hours(24))
    .setUpdateType(UpdateType.OnReadAndWrite)
    .cleanupFullSnapshot()
    .build();
```

## 性能优化

### 如何解决数据倾斜？

1. **添加随机前缀**：打散热点 Key
2. **使用 rebalance()**：重新分配数据
3. **两阶段聚合**：先局部聚合再全局聚合

```java
// 两阶段聚合示例
stream.keyBy(event -> event.getKey() + "_" + random.nextInt(10))
      .window(...)
      .reduce(...)
      .keyBy(event -> event.getKey().split("_")[0])
      .window(...)
      .reduce(...);
```

### 如何提高吞吐量？

1. 增加并行度
2. 使用本地聚合（combiner）
3. 启用对象重用
4. 优化序列化

```java
env.getConfig().enableObjectReuse();
```

## 部署运维

### 如何选择部署模式？

| 模式        | 场景     | 优点         |
| ----------- | -------- | ------------ |
| Session     | 开发测试 | 启动快       |
| Per-Job     | 生产环境 | 隔离性好     |
| Application | 推荐生产 | 最优资源利用 |

### 作业失败后如何恢复？

1. 检查是否有可用的检查点
2. 分析失败原因
3. 从最新检查点恢复

```bash
flink run -s hdfs:///checkpoints/<checkpoint-id> myJob.jar
```

## 常见错误

### ClassNotFoundException

**原因**：依赖未正确打包  
**解决**：检查 Maven shade 插件配置，确保依赖被打入 JAR

### OutOfMemoryError

**原因**：内存配置不足  
**解决**：增加 TaskManager 内存或减少并行度

### Checkpoint 超时

**原因**：状态过大或网络慢  
**解决**：

- 增大检查点超时时间
- 使用增量检查点
- 优化状态大小
