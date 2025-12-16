---
sidebar_position: 5
title: "DataStream API"
description: "Flink DataStream API 流处理编程详解"
---

# DataStream API

> 适用版本：Apache Flink v2.2.0

## 概述

DataStream API 是 Flink 的核心流处理 API，提供了丰富的算子用于处理无界和有界数据流。

## 执行环境

### 创建环境

```java
// 标准方式（自动检测本地或集群）
StreamExecutionEnvironment env =
    StreamExecutionEnvironment.getExecutionEnvironment();

// 本地环境（用于测试）
StreamExecutionEnvironment env =
    StreamExecutionEnvironment.createLocalEnvironment();

// 远程环境
StreamExecutionEnvironment env =
    StreamExecutionEnvironment.createRemoteEnvironment(
        "host", 8081, "path/to/jar");
```

### 配置环境

```java
// 设置并行度
env.setParallelism(4);

// 时间语义与水印：在 Source 上分配 WatermarkStrategy（无需显式设置 TimeCharacteristic）
// 例如：
// DataStream<Event> events = env.fromSource(
//     source,
//     WatermarkStrategy
//         .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(5))
//         .withTimestampAssigner((e, ts) -> e.getTimestamp()),
//     "my-source"
// );

// 设置重启策略
env.setRestartStrategy(RestartStrategies.fixedDelayRestart(3, Time.seconds(10)));
```

## 数据源（Source）

### 内置数据源

```java
// 从集合创建
DataStream<String> stream = env.fromElements("a", "b", "c");
DataStream<Integer> numbers = env.fromCollection(Arrays.asList(1, 2, 3));

// 从文件读取
DataStream<String> lines = env.readTextFile("/path/to/file");

// 从 Socket 读取
DataStream<String> socket = env.socketTextStream("localhost", 9999);

// 生成序列
DataStream<Long> sequence = env.fromSequence(1, 100);
```

### 自定义数据源

```java
public class MySource implements SourceFunction<String> {
    private volatile boolean isRunning = true;

    @Override
    public void run(SourceContext<String> ctx) throws Exception {
        while (isRunning) {
            ctx.collect("data-" + System.currentTimeMillis());
            Thread.sleep(1000);
        }
    }

    @Override
    public void cancel() {
        isRunning = false;
    }
}

// 使用自定义源
DataStream<String> stream = env.addSource(new MySource());
```

## 转换算子

### Map

一对一转换：

```java
DataStream<Integer> doubled = stream.map(x -> x * 2);

// 使用 MapFunction
DataStream<String> result = stream.map(new MapFunction<Integer, String>() {
    @Override
    public String map(Integer value) {
        return "Value: " + value;
    }
});
```

### FlatMap

一对多转换：

```java
DataStream<String> words = lines.flatMap(
    (String line, Collector<String> out) -> {
        for (String word : line.split(" ")) {
            out.collect(word);
        }
    }
).returns(Types.STRING); // 需要指定返回类型
```

### Filter

过滤数据：

```java
DataStream<Integer> positives = numbers.filter(x -> x > 0);
```

### KeyBy

按键分组：

```java
// 使用 Lambda
KeyedStream<Event, String> keyed = events.keyBy(event -> event.getKey());

// 使用字段名（POJO）
KeyedStream<Event, Tuple> keyed = events.keyBy("userId");

// 使用字段位置（Tuple）
KeyedStream<Tuple2<String, Integer>, Tuple> keyed = tuples.keyBy(0);
```

### Reduce

聚合操作：

```java
DataStream<Event> reduced = keyed.reduce(
    (e1, e2) -> new Event(e1.getKey(), e1.getValue() + e2.getValue())
);
```

### Aggregations

内置聚合：

```java
keyedStream.sum(1);           // 求和
keyedStream.min("field");     // 最小值
keyedStream.max("field");     // 最大值
keyedStream.minBy("field");   // 最小值对应的整条记录
keyedStream.maxBy("field");   // 最大值对应的整条记录
```

## 窗口操作

### 时间窗口

```java
// 滚动事件时间窗口
stream
    .keyBy(event -> event.getKey())
    .window(TumblingEventTimeWindows.of(Time.seconds(30)))
    .sum("value");

// 滑动处理时间窗口
stream
    .keyBy(event -> event.getKey())
    .window(SlidingProcessingTimeWindows.of(Time.minutes(5), Time.minutes(1)))
    .reduce((a, b) -> a.getValue() > b.getValue() ? a : b);
```

### 计数窗口

```java
// 滚动计数窗口
stream
    .keyBy(event -> event.getKey())
    .countWindow(100)
    .sum("value");

// 滑动计数窗口
stream
    .keyBy(event -> event.getKey())
    .countWindow(100, 10)
    .sum("value");
```

### 窗口函数

```java
// ProcessWindowFunction - 访问窗口元数据
stream
    .keyBy(event -> event.getKey())
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .process(new ProcessWindowFunction<Event, Result, String, TimeWindow>() {
        @Override
        public void process(String key, Context ctx,
                Iterable<Event> elements, Collector<Result> out) {
            long count = 0;
            for (Event e : elements) {
                count++;
            }
            out.collect(new Result(key, ctx.window().getEnd(), count));
        }
    });
```

## 多流操作

### Union

合并同类型流：

```java
DataStream<Event> unified = stream1.union(stream2, stream3);
```

### Connect

连接不同类型流：

```java
ConnectedStreams<String, Integer> connected =
    stringStream.connect(intStream);

DataStream<String> result = connected.map(
    new CoMapFunction<String, Integer, String>() {
        @Override
        public String map1(String value) {
            return "String: " + value;
        }

        @Override
        public String map2(Integer value) {
            return "Integer: " + value;
        }
    }
);
```

### Join

窗口 Join：

```java
stream1
    .join(stream2)
    .where(e -> e.getKey())
    .equalTo(e -> e.getKey())
    .window(TumblingEventTimeWindows.of(Time.seconds(10)))
    .apply((e1, e2) -> new JoinResult(e1, e2));
```

## 输出（Sink）

### 内置 Sink

```java
// 打印到控制台
stream.print();

// 写入文件
stream.writeAsText("/path/to/output");

// 写入 Socket
stream.writeToSocket("localhost", 9999, new SimpleStringSchema());
```

### 自定义 Sink

```java
public class MySink implements SinkFunction<String> {
    @Override
    public void invoke(String value, Context context) {
        System.out.println("Output: " + value);
    }
}

stream.addSink(new MySink());
```

## 执行作业

```java
// 执行并等待结果
JobExecutionResult result = env.execute("My Job");

// 异步执行
JobClient client = env.executeAsync("My Job");
JobExecutionResult result = client.getJobExecutionResult().get();
```

## 下一步

- 📊 [Table API & SQL](/docs/flink/table-sql) - 声明式数据处理
- 🎯 [核心概念](/docs/flink/core-concepts) - 深入理解 Flink 概念
- 🔧 [状态管理](/docs/flink/state-management) - 有状态计算详解
