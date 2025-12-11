---
sidebar_position: 11
title: "CEP 复杂事件处理"
description: "Flink CEP 复杂事件处理库详解"
---

# Flink CEP 复杂事件处理

## 概述

Flink CEP（Complex Event Processing）是 Flink 提供的复杂事件处理库，用于在事件流中检测符合特定模式的事件序列。

## 添加依赖

```xml
<dependency>
    <groupId>org.apache.flink</groupId>
    <artifactId>flink-cep</artifactId>
    <version>${flink.version}</version>
</dependency>
```

## 基本模式

### 简单模式

```java
// 匹配类型为 "start" 的事件
Pattern<Event, ?> pattern = Pattern.<Event>begin("start")
    .where(SimpleCondition.of(event -> event.getType().equals("start")));
```

### 模式序列

```java
// 匹配 start -> middle -> end 的事件序列
Pattern<Event, ?> pattern = Pattern.<Event>begin("start")
    .where(SimpleCondition.of(e -> e.getType().equals("start")))
    .next("middle")
    .where(SimpleCondition.of(e -> e.getType().equals("middle")))
    .followedBy("end")
    .where(SimpleCondition.of(e -> e.getType().equals("end")));
```

## 模式操作

### 量词

```java
// 匹配一次或多次
pattern.oneOrMore()

// 匹配指定次数
pattern.times(3)

// 匹配 2-4 次
pattern.times(2, 4)

// 匹配零次或多次
pattern.timesOrMore(2)

// 可选模式
pattern.optional()

// 贪婪模式
pattern.oneOrMore().greedy()
```

### 条件

```java
// 简单条件
.where(SimpleCondition.of(event -> event.getValue() > 100))

// 迭代条件（访问之前匹配的事件）
.where(new IterativeCondition<Event>() {
    @Override
    public boolean filter(Event event, Context<Event> ctx) throws Exception {
        for (Event prev : ctx.getEventsForPattern("start")) {
            if (event.getValue() > prev.getValue()) {
                return true;
            }
        }
        return false;
    }
})

// 组合条件
.where(condition1).or(condition2)
.where(condition1).and(condition2)
```

### 邻近策略

| 策略                | 描述               | 示例               |
| ------------------- | ------------------ | ------------------ |
| **next()**          | 严格邻近，必须紧邻 | A next B           |
| **followedBy()**    | 宽松邻近，允许间隔 | A ... B            |
| **followedByAny()** | 非确定性宽松邻近   | A ... B (多次匹配) |
| **notNext()**       | 严格不邻近         | A !next B          |
| **notFollowedBy()** | 宽松不邻近         | A !... B           |

```java
// 严格邻近：A 必须紧邻 B
Pattern<Event, ?> strict = Pattern.<Event>begin("a")
    .where(...)
    .next("b")
    .where(...);

// 宽松邻近：A 和 B 之间可以有其他事件
Pattern<Event, ?> relaxed = Pattern.<Event>begin("a")
    .where(...)
    .followedBy("b")
    .where(...);

// 不匹配：A 后面不能紧邻 B
Pattern<Event, ?> notPattern = Pattern.<Event>begin("a")
    .where(...)
    .notNext("b")
    .where(...);
```

### 时间约束

```java
// 整个模式必须在 10 分钟内完成
pattern.within(Time.minutes(10));

// 或使用 Duration
pattern.within(Duration.ofMinutes(10));
```

## 模式检测

### 应用模式

```java
DataStream<Event> input = ...;

// 定义模式
Pattern<Event, ?> pattern = Pattern.<Event>begin("start")
    .where(SimpleCondition.of(e -> e.getType().equals("login")))
    .next("middle")
    .where(SimpleCondition.of(e -> e.getType().equals("browse")))
    .followedBy("end")
    .where(SimpleCondition.of(e -> e.getType().equals("purchase")))
    .within(Time.minutes(30));

// 应用模式到数据流
PatternStream<Event> patternStream = CEP.pattern(
    input.keyBy(Event::getUserId),
    pattern
);

// 选择匹配的事件
DataStream<Alert> result = patternStream.process(
    new PatternProcessFunction<Event, Alert>() {
        @Override
        public void processMatch(Map<String, List<Event>> match,
                Context ctx, Collector<Alert> out) {
            Event login = match.get("start").get(0);
            Event purchase = match.get("end").get(0);
            out.collect(new Alert(login.getUserId(),
                "User completed purchase flow"));
        }
    }
);
```

### 处理超时

```java
// 定义超时输出标签
OutputTag<Event> timeoutTag = new OutputTag<Event>("timeout"){};

SingleOutputStreamOperator<Alert> result = patternStream.process(
    new PatternProcessFunction<Event, Alert>() {
        @Override
        public void processMatch(Map<String, List<Event>> match,
                Context ctx, Collector<Alert> out) {
            // 处理匹配
        }
    },
    new PatternTimeoutFunction<Event, Event>() {
        @Override
        public Event timeout(Map<String, List<Event>> match, long timestamp) {
            // 处理超时
            return match.get("start").get(0);
        }
    }
);

// 获取超时事件
DataStream<Event> timeoutStream = result.getSideOutput(timeoutTag);
```

## 实际案例

### 欺诈检测

检测短时间内多次失败登录后的成功登录：

```java
Pattern<LoginEvent, ?> fraudPattern = Pattern.<LoginEvent>begin("failed")
    .where(SimpleCondition.of(e -> !e.isSuccess()))
    .timesOrMore(3)
    .followedBy("success")
    .where(SimpleCondition.of(e -> e.isSuccess()))
    .within(Time.minutes(5));

CEP.pattern(loginStream.keyBy(LoginEvent::getUserId), fraudPattern)
    .process(new PatternProcessFunction<LoginEvent, FraudAlert>() {
        @Override
        public void processMatch(Map<String, List<LoginEvent>> match,
                Context ctx, Collector<FraudAlert> out) {
            List<LoginEvent> failedAttempts = match.get("failed");
            LoginEvent successLogin = match.get("success").get(0);
            out.collect(new FraudAlert(
                successLogin.getUserId(),
                failedAttempts.size(),
                "Suspicious login after multiple failures"
            ));
        }
    });
```

### 订单超时检测

检测创建后 15 分钟内未支付的订单：

```java
Pattern<OrderEvent, ?> timeoutPattern = Pattern.<OrderEvent>begin("create")
    .where(SimpleCondition.of(e -> e.getType().equals("create")))
    .followedBy("pay")
    .where(SimpleCondition.of(e -> e.getType().equals("pay")))
    .within(Time.minutes(15));

OutputTag<OrderEvent> timeoutTag = new OutputTag<OrderEvent>("timeout"){};

SingleOutputStreamOperator<OrderResult> result = CEP.pattern(
        orderStream.keyBy(OrderEvent::getOrderId),
        timeoutPattern
    )
    .process(new PatternProcessFunction<OrderEvent, OrderResult>() {
        @Override
        public void processMatch(Map<String, List<OrderEvent>> match,
                Context ctx, Collector<OrderResult> out) {
            out.collect(new OrderResult(
                match.get("create").get(0).getOrderId(),
                "PAID"
            ));
        }
    });

// 获取超时订单
DataStream<OrderEvent> timeoutOrders = result.getSideOutput(timeoutTag);
```

### 温度异常检测

检测温度在短时间内急剧变化：

```java
Pattern<SensorReading, ?> warningPattern = Pattern.<SensorReading>begin("first")
    .next("second")
    .where(new IterativeCondition<SensorReading>() {
        @Override
        public boolean filter(SensorReading current, Context<SensorReading> ctx) {
            SensorReading first = ctx.getEventsForPattern("first")
                .iterator().next();
            return Math.abs(current.getTemperature() - first.getTemperature()) > 10;
        }
    })
    .within(Time.seconds(10));
```

## 最佳实践

### 性能优化

1. **合理设置时间窗口**：过长的 within 会增加状态大小
2. **使用 keyBy**：确保按业务键分区，避免状态膨胀
3. **及时清理状态**：使用 TTL 或手动清理

### 注意事项

- CEP 模式会产生状态，需要配合检查点使用
- 复杂模式可能消耗大量内存
- 模式中的 `notFollowedBy()` 不能作为结尾

## 下一步

- 🔌 [连接器](/docs/flink/connectors) - 数据源与接收器
- 🚀 [性能优化](/docs/flink/performance-optimization) - 调优指南
- 📋 [最佳实践](/docs/flink/best-practices) - 开发规范
