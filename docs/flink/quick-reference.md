---
sidebar_position: 7
title: "快速参考"
description: "Flink 常用配置和命令速查"
---

# Flink 快速参考

> 适用版本：Apache Flink v2.2.0

## 常用命令

### 集群管理

```bash
# 启动/停止集群
./bin/start-cluster.sh
./bin/stop-cluster.sh

# 启动/停止 JobManager
./bin/jobmanager.sh start
./bin/jobmanager.sh stop

# 启动/停止 TaskManager
./bin/taskmanager.sh start
./bin/taskmanager.sh stop
```

### 作业管理

```bash
# 提交作业
flink run -c com.example.MyJob myJob.jar
flink run -p 4 myJob.jar                    # 并行度为 4
flink run -d myJob.jar                       # 后台运行
flink run -s /path/to/savepoint myJob.jar   # 从保存点恢复

# 查看作业
flink list                                   # 运行中的作业
flink list -a                                # 所有作业

# 取消作业
flink cancel <jobId>
flink cancel -s /path/to/savepoint <jobId>  # 取消并保存点

# 保存点
flink savepoint <jobId> /path/to/savepoints
```

## 常用配置

### flink-conf.yaml

```yaml
# JobManager 配置
jobmanager.rpc.address: localhost
jobmanager.rpc.port: 6123
jobmanager.memory.process.size: 1600m

# TaskManager 配置
taskmanager.memory.process.size: 4096m
taskmanager.numberOfTaskSlots: 4

# 并行度
parallelism.default: 4

# 检查点
execution.checkpointing.interval: 60000
state.backend: rocksdb
state.checkpoints.dir: hdfs:///checkpoints

# Web UI
rest.port: 8081
```

## API 速查

### DataStream 常用操作

| 操作    | 示例                                                                       |
| ------- | -------------------------------------------------------------------------- |
| map     | `stream.map(x -> x * 2)`                                                   |
| flatMap | `stream.flatMap((s, c) -> { for(String w : s.split(" ")) c.collect(w); })` |
| filter  | `stream.filter(x -> x > 0)`                                                |
| keyBy   | `stream.keyBy(e -> e.getKey())`                                            |
| reduce  | `keyed.reduce((a, b) -> a + b)`                                            |
| sum     | `keyed.sum("field")`                                                       |
| window  | `keyed.window(TumblingEventTimeWindows.of(Time.seconds(5)))`               |
| union   | `stream1.union(stream2)`                                                   |
| connect | `stream1.connect(stream2)`                                                 |

### 窗口类型

| 窗口             | 创建方式                                                      |
| ---------------- | ------------------------------------------------------------- |
| 滚动事件时间窗口 | `TumblingEventTimeWindows.of(Time.hours(1))`                  |
| 滑动事件时间窗口 | `SlidingEventTimeWindows.of(Time.hours(1), Time.minutes(15))` |
| 会话窗口         | `EventTimeSessionWindows.withGap(Time.minutes(10))`           |
| 滚动处理时间窗口 | `TumblingProcessingTimeWindows.of(Time.hours(1))`             |
| 计数窗口         | `countWindow(100)`                                            |

### 状态类型

| 状态          | 描述 | 初始化                                                       |
| ------------- | ---- | ------------------------------------------------------------ |
| ValueState    | 单值 | `new ValueStateDescriptor<>("name", Type.class)`             |
| ListState     | 列表 | `new ListStateDescriptor<>("name", Type.class)`              |
| MapState      | 映射 | `new MapStateDescriptor<>("name", Key.class, Value.class)`   |
| ReducingState | 聚合 | `new ReducingStateDescriptor<>("name", reducer, Type.class)` |

## SQL 速查

### 常用 DDL

```sql
-- 创建表
CREATE TABLE table_name (
    column_name TYPE,
    ts TIMESTAMP(3),
    WATERMARK FOR ts AS ts - INTERVAL '5' SECOND,
    PRIMARY KEY (id) NOT ENFORCED
) WITH (...);

-- 删除表
DROP TABLE IF EXISTS table_name;
```

### 常用函数

| 函数        | 描述       | 示例                            |
| ----------- | ---------- | ------------------------------- |
| CONCAT      | 字符串连接 | `CONCAT(a, b)`                  |
| UPPER/LOWER | 大小写     | `UPPER(name)`                   |
| SUBSTRING   | 截取       | `SUBSTRING(str, 1, 5)`          |
| CAST        | 类型转换   | `CAST(x AS STRING)`             |
| COALESCE    | 空值处理   | `COALESCE(a, b, 'default')`     |
| DATE_FORMAT | 日期格式化 | `DATE_FORMAT(ts, 'yyyy-MM-dd')` |

### 窗口函数

```sql
-- 滚动窗口
TUMBLE(time_col, INTERVAL '1' HOUR)

-- 滑动窗口
HOP(time_col, INTERVAL '5' MINUTE, INTERVAL '1' HOUR)

-- 会话窗口
SESSION(time_col, INTERVAL '30' MINUTE)
```

## 常见问题快速解决

| 问题         | 解决方案                          |
| ------------ | --------------------------------- |
| 作业提交失败 | 检查集群状态、检查 JAR 包         |
| 内存不足     | 增加 TaskManager 内存或减少并行度 |
| 检查点失败   | 检查状态后端配置、增大超时时间    |
| 数据倾斜     | 添加随机前缀、使用 rebalance()    |
| 延迟数据     | 使用侧输出、增大水印延迟          |
