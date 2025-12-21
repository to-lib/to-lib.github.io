---
sidebar_position: 17
title: "最佳实践"
description: "Flink 开发与生产最佳实践"
---

# Flink 最佳实践

> 适用版本：Apache Flink v2.2.0

## 代码开发

### 算子设计

```java
// ✅ 推荐：使用 RichFunction 访问运行时上下文
public class MyRichMapFunction extends RichMapFunction<Event, Result> {
    private transient Counter counter;

    @Override
    public void open(Configuration parameters) {
        counter = getRuntimeContext()
            .getMetricGroup()
            .counter("processedEvents");
    }

    @Override
    public Result map(Event event) {
        counter.inc();
        return process(event);
    }
}

// ❌ 避免：在算子中创建不可序列化的对象
public class BadMapFunction implements MapFunction<Event, Result> {
    private Connection connection; // 不可序列化！
}
```

### 状态使用

```java
// ✅ 推荐：在 open() 中初始化状态
@Override
public void open(Configuration parameters) {
    ValueStateDescriptor<Long> descriptor =
        new ValueStateDescriptor<>("count", Long.class);
    countState = getRuntimeContext().getState(descriptor);
}

// ✅ 推荐：使用状态 TTL 防止状态无限增长
StateTtlConfig ttlConfig = StateTtlConfig
    .newBuilder(Time.days(7))
    .setUpdateType(UpdateType.OnCreateAndWrite)
    .cleanupIncrementally(10, true)
    .build();
descriptor.enableTimeToLive(ttlConfig);
```

### 时间处理

```java
// ✅ 推荐：使用事件时间（无需显式设置 TimeCharacteristic）
// ✅ 推荐：正确设置水印
DataStream<Event> stream = source.assignTimestampsAndWatermarks(
    WatermarkStrategy
        .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(5))
        .withTimestampAssigner((event, ts) -> event.getTimestamp())
        .withIdleness(Duration.ofMinutes(1))  // 处理空闲分区
);
```

### 异常处理

```java
// ✅ 推荐：使用侧输出处理异常数据
OutputTag<Event> errorTag = new OutputTag<Event>("errors"){};

SingleOutputStreamOperator<Result> result = stream
    .process(new ProcessFunction<Event, Result>() {
        @Override
        public void processElement(Event event, Context ctx,
                Collector<Result> out) {
            try {
                out.collect(process(event));
            } catch (Exception e) {
                ctx.output(errorTag, event);
            }
        }
    });

// 获取异常数据
DataStream<Event> errors = result.getSideOutput(errorTag);
```

## 生产配置

### 检查点配置

```java
// 生产环境推荐配置
env.enableCheckpointing(60000);  // 1 分钟

CheckpointConfig config = env.getCheckpointConfig();
config.setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
config.setMinPauseBetweenCheckpoints(30000);  // 最小间隔
config.setCheckpointTimeout(600000);  // 10 分钟超时
config.setMaxConcurrentCheckpoints(1);
config.setExternalizedCheckpointCleanup(
    ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);

// 对于大状态，使用非对齐检查点
config.enableUnalignedCheckpoints();
```

### 重启策略

```java
// 固定延迟重启
env.setRestartStrategy(RestartStrategies.fixedDelayRestart(
    3,  // 最多重启 3 次
    Time.seconds(30)  // 重启间隔
));

// 失败率重启
env.setRestartStrategy(RestartStrategies.failureRateRestart(
    3,  // 时间窗口内最大失败次数
    Time.minutes(5),  // 时间窗口
    Time.seconds(10)  // 重启间隔
));

// 指数延迟重启
env.setRestartStrategy(RestartStrategies.exponentialDelayRestart(
    Time.seconds(1),  // 初始延迟
    Time.minutes(5),  // 最大延迟
    2.0,  // 延迟倍数
    Time.hours(1),  // 重置窗口
    0.1  // 抖动因子
));
```

### 资源配置

```yaml
# 生产环境 flink-conf.yaml
jobmanager.memory.process.size: 4096m
taskmanager.memory.process.size: 8192m
taskmanager.numberOfTaskSlots: 4

# 状态后端
state.backend: rocksdb
state.backend.incremental: true
state.checkpoints.dir: hdfs:///flink/checkpoints
state.savepoints.dir: hdfs:///flink/savepoints

# 高可用
high-availability: zookeeper
high-availability.storageDir: hdfs:///flink/ha
high-availability.zookeeper.quorum: zk1:2181,zk2:2181,zk3:2181
```

## 监控告警

### 关键监控指标

```yaml
# Prometheus 告警规则
groups:
  - name: flink-alerts
    rules:
      - alert: FlinkJobFailed
        expr: flink_jobmanager_job_uptime == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Flink job failed"

      - alert: FlinkCheckpointFailed
        expr: increase(flink_jobmanager_job_numberOfFailedCheckpoints[5m]) > 0
        for: 5m
        labels:
          severity: warning

      - alert: FlinkHighBackpressure
        expr: flink_taskmanager_job_task_isBackPressured > 0.5
        for: 10m
        labels:
          severity: warning
```

### 日志规范

```java
// ✅ 推荐：使用 SLF4J 并避免在热路径打印日志
private static final Logger LOG = LoggerFactory.getLogger(MyFunction.class);

@Override
public void processElement(Event event, Context ctx, Collector<Result> out) {
    // 避免在每条记录上打印日志
    if (LOG.isDebugEnabled()) {
        LOG.debug("Processing event: {}", event.getId());
    }
}
```

## 版本升级

### 升级步骤

1. **创建保存点**

   ```bash
   flink savepoint <jobId> hdfs:///savepoints
   ```

2. **取消旧作业**

   ```bash
   flink cancel <jobId>
   ```

3. **部署新版本**

4. **从保存点恢复**
   ```bash
   flink run -s hdfs:///savepoints/savepoint-xxx newJob.jar
   ```

### 状态兼容性

```java
// ✅ 推荐：使用明确的状态名称
ValueStateDescriptor<Long> descriptor =
    new ValueStateDescriptor<>("counter-v1", Long.class);

// ✅ 推荐：使用 Avro/Protobuf 等可演进的序列化格式
```

## 测试建议

### 单元测试

```java
@Test
public void testMapFunction() throws Exception {
    MyMapFunction function = new MyMapFunction();
    function.open(new Configuration());

    Result result = function.map(new Event("test", 100));

    assertEquals("expected", result.getValue());
}
```

### 集成测试

```java
@Test
public void testPipeline() throws Exception {
    StreamExecutionEnvironment env =
        StreamExecutionEnvironment.getExecutionEnvironment();

    DataStream<Event> input = env.fromElements(
        new Event("a", 1),
        new Event("b", 2)
    );

    DataStream<Result> output = MyPipeline.build(input);

    List<Result> results = new ArrayList<>();
    output.executeAndCollect().forEachRemaining(results::add);

    assertEquals(2, results.size());
}
```

### 使用 MiniCluster

```java
@ClassRule
public static MiniClusterResource flinkCluster =
    new MiniClusterResource(
        new MiniClusterResourceConfiguration.Builder()
            .setNumberSlotsPerTaskManager(2)
            .setNumberTaskManagers(1)
            .build()
    );
```

## 常见反模式

| 反模式         | 问题                | 解决方案                       |
| -------------- | ------------------- | ------------------------------ |
| 算子中创建连接 | 不可序列化/资源泄漏 | 使用 RichFunction + open/close |
| 热路径打印日志 | 性能下降            | 使用采样或 debug 级别          |
| 忽略背压       | 延迟增加            | 监控并优化                     |
| 无限状态增长   | OOM                 | 使用状态 TTL                   |
| 不设置水印     | 窗口不触发          | 正确配置水印策略               |

## 下一步

- 🔧 [部署与运维](/docs/flink/deployment) - 生产部署
- 🚀 [性能优化](/docs/flink/performance-optimization) - 调优指南
- 💼 [面试题精选](/docs/interview/flink-interview-questions) - 面试准备
