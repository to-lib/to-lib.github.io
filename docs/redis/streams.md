---
sidebar_position: 11
title: Stream 数据流
---

# Redis Stream 数据流

Redis Stream 是 Redis 5.0 引入的新数据类型，专门用于消息队列和日志存储场景。

## Stream 简介

### 核心概念

- **Stream** - 消息流，类似于只能追加的日志
- **Entry** - 消息条目，包含 ID 和字段-值对
- **Consumer Group** - 消费组，支持分布式消费
- **Consumer** - 消费者，从 Stream 读取消息

### 与 Pub/Sub 对比

| 特性     | Stream    | Pub/Sub     |
| -------- | --------- | ----------- |
| 持久化   | ✅ 持久化 | ❌ 不持久化 |
| 历史消息 | ✅ 可读取 | ❌ 不可读取 |
| 消费组   | ✅ 支持   | ❌ 不支持   |
| ACK 确认 | ✅ 支持   | ❌ 不支持   |
| 消息重试 | ✅ 支持   | ❌ 不支持   |

## 基本命令

### XADD - 添加消息

```bash
# 语法
XADD stream_name ID field1 value1 [field2 value2 ...]

# 示例：自动生成 ID
XADD mystream * name "Alice" age "25"
# 返回：1609459200000-0

# 指定 ID
XADD mystream 1609459200000-0 name "Bob" age "30"

# ID 格式：毫秒时间戳-序列号
```

### XREAD - 读取消息

```bash
# 读取最新消息
XREAD COUNT 10 STREAMS mystream 0

# 阻塞读取
XREAD BLOCK 5000 STREAMS mystream $

# 从指定 ID 开始读取
XREAD STREAMS mystream 1609459200000-0
```

### XLEN - 获取长度

```bash
# 获取 Stream 中消息数量
XLEN mystream
```

### XRANGE - 范围查询

```bash
# 查询所有消息
XRANGE mystream - +

# 查询指定范围
XRANGE mystream 1609459200000-0 1609459300000-0

# 限制数量
XRANGE mystream - + COUNT 10
```

### XDEL - 删除消息

```bash
# 删除指定消息
XDEL mystream 1609459200000-0
```

### XTRIM - 裁剪 Stream

```bash
# 保留最新 1000 条
XTRIM mystream MAXLEN 1000

# 近似裁剪（性能更好）
XTRIM mystream MAXLEN ~ 1000
```

## 消费组

### XGROUP CREATE - 创建消费组

```bash
# 语法
XGROUP CREATE stream_name group_name start_id

# 从头开始消费
XGROUP CREATE mystream mygroup 0

# 从stream末尾开始
XGROUP CREATE mystream mygroup $

# 创建stream并创建消费组
XGROUP CREATE mystream mygroup $ MKSTREAM
```

### XREADGROUP - 组内消费

```bash
# 语法
XREADGROUP GROUP group consumer COUNT count STREAMS stream >

# 示例
XREADGROUP GROUP mygroup consumer1 COUNT 10 STREAMS mystream >

# 阻塞读取
XREADGROUP GROUP mygroup consumer1 BLOCK 5000 COUNT 10 STREAMS mystream >
```

### XACK - 确认消息

```bash
# 确认已处理的消息
XACK mystream mygroup 1609459200000-0 1609459200001-0
```

### XPENDING - 查看待确认消息

```bash
# 查看消费组的待确认消息
XPENDING mystream mygroup

# 查看详细信息
XPENDING mystream mygroup - + 10

# 查看指定消费者的待确认消息
XPENDING mystream mygroup - + 10 consumer1
```

### XCLAIM - 消息转移

```bash
# 将消息转移给其他消费者
XCLAIM mystream mygroup consumer2 3600000 1609459200000-0
```

## Java 实现

### 添加消息

```java
import redis.clients.jedis.Jedis;
import redis.clients.jedis.StreamEntryID;

public class StreamProducer {
    public static void main(String[] args) {
        Jedis jedis = new Jedis("localhost", 6379);

        // 添加消息
        Map<String, String> message = new HashMap<>();
        message.put("user", "Alice");
        message.put("action", "login");
        message.put("timestamp", String.valueOf(System.currentTimeMillis()));

        StreamEntryID id = jedis.xadd(
            "user-events",      // stream name
            StreamEntryID.NEW_ENTRY,  // 自动生成ID
            message
        );

        System.out.println("消息ID: " + id);
        jedis.close();
    }
}
```

### 简单消费

```java
import redis.clients.jedis.Jedis;
import redis.clients.jedis.StreamEntry;
import redis.clients.jedis.params.XReadParams;

import java.util.List;
import java.util.Map;

public class StreamConsumer {
    public static void main(String[] args) {
        Jedis jedis = new Jedis("localhost", 6379);

        String lastId = "0";  // 从头开始

        while (true) {
            // 读取消息
            List<Map.Entry<String, List<StreamEntry>>> result = jedis.xread(
                XReadParams.xReadParams().count(10).block(5000),
                Map.of("user-events", lastId)
            );

            if (result != null && !result.isEmpty()) {
                for (Map.Entry<String, List<StreamEntry>> entry : result) {
                    for (StreamEntry streamEntry : entry.getValue()) {
                        System.out.println("消息ID: " + streamEntry.getID());
                        System.out.println("内容: " + streamEntry.getFields());

                        lastId = streamEntry.getID().toString();
                    }
                }
            }
        }
    }
}
```

### 消费组实现

```java
import redis.clients.jedis.Jedis;
import redis.clients.jedis.StreamEntry;
import redis.clients.jedis.params.XReadGroupParams;

public class StreamGroupConsumer {
    public static void main(String[] args) {
        Jedis jedis = new Jedis("localhost", 6379);

        String streamName = "user-events";
        String groupName = "event-processors";
        String consumerName = "consumer-1";

        try {
            // 创建消费组
            jedis.xgroupCreate(streamName, groupName, "0", true);
        } catch (Exception e) {
            // 消费组已存在
        }

        while (true) {
            // 读取消息
            List<Map.Entry<String, List<StreamEntry>>> messages =
                jedis.xreadGroup(
                    groupName,
                    consumerName,
                    XReadGroupParams.xReadGroupParams().count(10).block(5000),
                    Map.of(streamName, ">")
                );

            if (messages != null && !messages.isEmpty()) {
                for (Map.Entry<String, List<StreamEntry>> entry : messages) {
                    for (StreamEntry msg : entry.getValue()) {
                        try {
                            // 处理消息
                            processMessage(msg);

                            // 确认消息
                            jedis.xack(streamName, groupName, msg.getID());
                        } catch (Exception e) {
                            System.err.println("处理失败: " + e.getMessage());
                        }
                    }
                }
            }
        }
    }

    private static void processMessage(StreamEntry msg) {
        System.out.println("处理消息: " + msg.getID());
        System.out.println("内容: " + msg.getFields());
    }
}
```

### Spring Boot 集成

```java
@Configuration
public class RedisStreamConfig {

    @Bean
    public StreamMessageListenerContainer<String, MapRecord<String, String, String>>
            streamMessageListenerContainer(RedisConnectionFactory connectionFactory) {

        StreamMessageListenerContainerOptions<String, MapRecord<String, String, String>> options =
            StreamMessageListenerContainerOptions
                .builder()
                .pollTimeout(Duration.ofSeconds(1))
                .build();

        StreamMessageListenerContainer<String, MapRecord<String, String, String>> container =
            StreamMessageListenerContainer.create(connectionFactory, options);

        // 订阅 Stream
        container.receive(
            Consumer.from("event-group", "consumer-1"),
            StreamOffset.create("user-events", ReadOffset.lastConsumed()),
            message -> {
                System.out.println("收到消息: " + message.getValue());
                // 处理消息...
            }
        );

        container.start();
        return container;
    }
}
```

## 应用场景

### 1. 消息队列

```bash
# 生产者
XADD task-queue * type "email" to "user@example.com" subject "Welcome"

# 消费组
XGROUP CREATE task-queue email-workers $ MKSTREAM

# 消费者
XREADGROUP GROUP email-workers worker-1 COUNT 1 STREAMS task-queue >
```

### 2. 日志收集

```bash
# 应用日志
XADD app-logs * level "ERROR" message "Connection timeout" service "api-server"

# 访问日志
XADD access-logs * method "GET" path "/api/users" status "200" duration "45ms"

# 查询最近的错误日志
XREVRANGE app-logs + - COUNT 100
```

### 3. 实时监控

```bash
# 系统指标
XADD metrics * cpu "45.2" memory "78.5" disk "62.1"

# 业务指标
XADD business-metrics * orders "1234" revenue "56789.00" users "9876"
```

### 4. 事件溯源

```bash
# 记录所有事件
XADD order-events * type "created" orderId "12345" amount "99.99"
XADD order-events * type "paid" orderId "12345" paymentId "PAY-001"
XADD order-events * type "shipped" orderId "12345" trackingNo "TRACK-001"

# 回放事件
XRANGE order-events - +
```

## 高级特性

### 1. 消息重试

```java
// 查找长时间未确认的消息
List<StreamPendingEntry> pending = jedis.xpending(
    streamName,
    groupName,
    null, null, 100,
    consumerName
);

// 重新处理超时消息
for (StreamPendingEntry entry : pending) {
    if (entry.getIdleTime() > 60000) {  // 超过1分钟
        // 转移给当前消费者
        List<StreamEntry> claimed = jedis.xclaim(
            streamName,
            groupName,
            consumerName,
            60000,
            entry.getID()
        );

        // 重新处理
        for (StreamEntry msg : claimed) {
            processMessage(msg);
        }
    }
}
```

### 2. 消息过期

```bash
# 自动裁剪，保留最新 10000 条
XTRIM mystream MAXLEN ~ 10000

# 基于时间裁剪（需要定期执行）
# 删除 7 天前的消息
XDEL mystream $(XRANGE mystream - $(expr $(date +%s) - 604800)000-0)
```

### 3. 多 Stream 监听

```java
// 同时监听多个 Stream
Map<String, StreamEntryID> streams = new HashMap<>();
streams.put("orders", lastOrderId);
streams.put("payments", lastPaymentId);
streams.put("shipments", lastShipmentId);

List<Map.Entry<String, List<StreamEntry>>> results =
    jedis.xread(
        XReadParams.xReadParams().count(10).block(1000),
        streams
    );
```

## 性能优化

### 1. 批量操作

```bash
# Pipeline 批量添加
MULTI
XADD stream1 * field1 value1
XADD stream1 * field2 value2
XADD stream1 * field3 value3
EXEC
```

### 2. 合理设置 MAXLEN

```bash
# 避免无限增长
XADD mystream MAXLEN ~ 10000 * field value

# 定期裁剪
XTRIM mystream MAXLEN ~ 10000
```

### 3. 使用 COUNT 限制

```bash
# 避免一次读取过多
XREAD COUNT 100 STREAMS mystream 0
```

## 监控和管理

### XINFO - 查看信息

```bash
# Stream 信息
XINFO STREAM mystream

# 消费组信息
XINFO GROUPS mystream

# 消费者信息
XINFO CONSUMERS mystream mygroup
```

## 最佳实践

### 1. 合理划分消费组

```bash
# 按功能划分
XGROUP CREATE orders email-group $    # 发邮件
XGROUP CREATE orders sms-group $      # 发短信
XGROUP CREATE orders log-group $      # 记录日志
```

### 2. 消息格式化

```java
// 使用 JSON 格式
String message = new JSONObject()
    .put("eventType", "ORDER_CREATED")
    .put("orderId", "12345")
    .put("timestamp", System.currentTimeMillis())
    .put("data", new JSONObject()
        .put("amount", 99.99)
        .put("userId", 1001)
    )
    .toString();

jedis.xadd("orders", "*", Map.of("payload", message));
```

### 3. 错误处理

```java
try {
    processMessage(msg);
    jedis.xack(streamName, groupName, msg.getID());
} catch (Exception e) {
    // 记录错误
    log.error("处理失败", e);

    // 转移到死信队列
    jedis.xadd(
        "dlq-" + streamName,
        "*",
        Map.of(
            "originalId", msg.getID().toString(),
            "error", e.getMessage(),
            "payload", msg.getFields().toString()
        )
    );
}
```

### 4. 优雅关闭

```java
Runtime.getRuntime().addShutdownHook(new Thread(() -> {
    container.stop();
    jedis.close();
}));
```

## 总结

- ✅ Stream 是功能强大的消息队列解决方案
- ✅ 支持持久化、消费组、ACK 确认
- ✅ 适合日志收集、消息队列、事件溯源
- ✅ 比 Pub/Sub 更可靠，但开销稍大
- 💡 生产环境推荐使用 Stream 代替 Pub/Sub

Stream 是 Redis 消息队列的最佳选择！
