---
sidebar_position: 8
title: 发布订阅
---

# Redis 发布订阅

Redis 发布订阅（Pub/Sub）是一种消息通信模式，允许发送者（pub）发送消息，订阅者（sub）接收消息。

## 基本概念

### 发布订阅模式

发布订阅模式包含三个角色：

- **发布者（Publisher）** - 发送消息到频道
- **订阅者（Subscriber）** - 订阅频道接收消息
- **频道（Channel）** - 消息传输的通道

**特点：**

- 发布者和订阅者解耦
- 一个消息可以被多个订阅者接收
- 订阅者只能收到订阅后发布的消息

## 基本命令

### SUBSCRIBE - 订阅频道

```bash
# 订阅一个或多个频道
SUBSCRIBE channel1 channel2 channel3
```

### PUBLISH - 发布消息

```bash
# 向频道发布消息
PUBLISH channel1 "Hello World"

# 返回值：接收到消息的订阅者数量
```

### UNSUBSCRIBE - 取消订阅

```bash
# 取消订阅指定频道
UNSUBSCRIBE channel1

# 取消所有订阅
UNSUBSCRIBE
```

## 模式订阅

### PSUBSCRIBE - 模式订阅

```bash
# 使用通配符订阅多个频道
PSUBSCRIBE news.*

# 匹配规则
# * 匹配任意字符
# ? 匹配单个字符
# [abc] 匹配括号内任一字符
```

示例：

```bash
# 订阅所有 news 开头的频道
PSUBSCRIBE news.*

# 匹配：news.sports, news.tech, news.finance
```

### PUNSUBSCRIBE - 取消模式订阅

```bash
# 取消模式订阅
PUNSUBSCRIBE news.*

# 取消所有模式订阅
PUNSUBSCRIBE
```

## 实战示例

### 示例 1: 聊天室

**订阅者（客户端 1）：**

```bash
redis-cli
127.0.0.1:6379> SUBSCRIBE chatroom

# 输出
Reading messages... (press Ctrl-C to quit)
1) "subscribe"
2) "chatroom"
3) (integer) 1
```

**发布者（客户端 2）：**

```bash
redis-cli
127.0.0.1:6379> PUBLISH chatroom "Alice: Hello everyone!"
(integer) 1

127.0.0.1:6379> PUBLISH chatroom "Bob: Hi Alice!"
(integer) 1
```

**订阅者接收：**

```bash
1) "message"
2) "chatroom"
3) "Alice: Hello everyone!"

1) "message"
2) "chatroom"
3) "Bob: Hi Alice!"
```

### 示例 2: 新闻分类

```bash
# 订阅体育新闻
SUBSCRIBE news.sports

# 订阅科技新闻
SUBSCRIBE news.tech

# 发布体育新闻
PUBLISH news.sports "Lakers won the championship!"

# 发布科技新闻
PUBLISH news.tech "New iPhone released!"
```

### 示例 3: 日志收集

```bash
# 订阅所有日志
PSUBSCRIBE log.*

# 发布不同级别日志
PUBLISH log.error "Database connection failed"
PUBLISH log.warning "High memory usage detected"
PUBLISH log.info "User login successful"
```

## Java 实现

### 订阅者

```java
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPubSub;

public class RedisSubscriber {
    public static void main(String[] args) {
        Jedis jedis = new Jedis("localhost", 6379);

        // 创建订阅监听器
        JedisPubSub jedisPubSub = new JedisPubSub() {
            @Override
            public void onMessage(String channel, String message) {
                System.out.println("收到消息：" + channel + " -> " + message);
            }

            @Override
            public void onSubscribe(String channel, int subscribedChannels) {
                System.out.println("订阅频道：" + channel);
            }

            @Override
            public void onUnsubscribe(String channel, int subscribedChannels) {
                System.out.println("取消订阅：" + channel);
            }
        };

        // 订阅频道（阻塞）
        jedis.subscribe(jedisPubSub, "news", "sports");
    }
}
```

### 发布者

```java
import redis.clients.jedis.Jedis;

public class RedisPublisher {
    public static void main(String[] args) {
        Jedis jedis = new Jedis("localhost", 6379);

        // 发布消息
        Long receivers = jedis.publish("news", "Breaking news!");
        System.out.println("消息发送给 " + receivers + " 个订阅者");

        jedis.close();
    }
}
```

### Spring Boot 集成

```java
@Configuration
public class RedisConfig {

    @Bean
    RedisMessageListenerContainer container(
            RedisConnectionFactory connectionFactory,
            MessageListenerAdapter listenerAdapter) {

        RedisMessageListenerContainer container =
            new RedisMessageListenerContainer();
        container.setConnectionFactory(connectionFactory);

        // 订阅频道
        container.addMessageListener(
            listenerAdapter,
            new PatternTopic("news.*")
        );

        return container;
    }

    @Bean
    MessageListenerAdapter listenerAdapter(RedisMessageListener receiver) {
        return new MessageListenerAdapter(receiver, "receiveMessage");
    }
}

@Component
public class RedisMessageListener {

    public void receiveMessage(String message) {
        System.out.println("接收到消息：" + message);
    }
}
```

## 应用场景

### 1. 实时通知

```bash
# 用户关注通知
PUBLISH user:1001:notifications "新的关注者：user:2002"

# 系统通知
PUBLISH system:announcements "系统将于今晚 23:00 维护"
```

### 2. 消息推送

```bash
# 推送订单状态更新
PUBLISH order:updates "订单 #12345 已发货"

# 推送库存变化
PUBLISH inventory:changes "商品 SKU-001 库存不足"
```

### 3. 日志收集

```bash
# 应用日志
PUBLISH log:app:error "NullPointerException at line 123"
PUBLISH log:app:info "Application started successfully"

# 访问日志
PUBLISH log:access "GET /api/users - 200 OK"
```

### 4. 分布式事件通知

```bash
# 缓存失效通知
PUBLISH cache:invalidate "user:1001"

# 配置更新通知
PUBLISH config:update "database.pool.size=20"
```

## Pub/Sub vs Stream

| 特性       | Pub/Sub        | Stream           |
| ---------- | -------------- | ---------------- |
| 消息持久化 | ❌ 不持久化    | ✅ 持久化        |
| 历史消息   | ❌ 无法获取    | ✅ 可获取        |
| 消费组     | ❌ 不支持      | ✅ 支持          |
| 消息确认   | ❌ 无          | ✅ 支持 ACK      |
| 适用场景   | 实时通知、广播 | 消息队列、日志流 |

## 注意事项

### 1. 消息不持久化

```bash
# 订阅前发的消息收不到
PUBLISH channel1 "message1"  # 此时无订阅者

SUBSCRIBE channel1  # 现在订阅
# 收不到 message1
```

**解决方案：** 使用 Stream 或其他消息队列

### 2. 订阅是阻塞操作

```java
// 此操作会阻塞当前线程
jedis.subscribe(jedisPubSub, "channel1");

// 后续代码不会执行
System.out.println("This won't print");
```

**解决方案：** 在单独线程中订阅

```java
new Thread(() -> {
    jedis.subscribe(jedisPubSub, "channel1");
}).start();
```

### 3. 网络问题导致消息丢失

订阅者断开连接期间的消息会丢失。

**解决方案：**

- 使用心跳检测
- 实现重连机制
- 考虑使用 Stream

### 4. 无消息确认机制

发布者不知道消息是否被成功接收。

**解决方案：** 使用 Stream 的 ACK 机制

## 性能优化

### 1. 使用模式订阅要谨慎

```bash
# 避免过于宽泛的模式
PSUBSCRIBE *  # 不推荐，匹配所有频道

# 使用更具体的模式
PSUBSCRIBE news:tech:*  # 推荐
```

### 2. 控制订阅数量

```java
// 不要订阅过多频道
// 每个订阅都会消耗资源
jedis.subscribe(jedisPubSub,
    "ch1", "ch2", "ch3"  // 适量
);
```

### 3. 合理设置超时

```java
JedisPool pool = new JedisPool(
    new JedisPoolConfig(),
    "localhost",
    6379,
    2000  // 超时时间 2秒
);
```

## 监控命令

### PUBSUB CHANNELS

```bash
# 列出所有活跃频道
PUBSUB CHANNELS

# 列出匹配的频道
PUBSUB CHANNELS news.*
```

### PUBSUB NUMSUB

```bash
# 查看频道订阅者数量
PUBSUB NUMSUB channel1 channel2

# 输出
1) "channel1"
2) (integer) 3
3) "channel2"
4) (integer) 1
```

### PUBSUB NUMPAT

```bash
# 查看模式订阅数量
PUBSUB NUMPAT

# 输出
(integer) 5
```

## 最佳实践

### 1. 频道命名规范

```bash
# 使用层级结构
app:notifications:user:1001
log:level:error
event:type:order:created

# 便于模式订阅
PSUBSCRIBE app:notifications:*
PSUBSCRIBE log:level:*
```

### 2. 消息格式化

```java
// 使用 JSON 格式
String message = new JSONObject()
    .put("type", "notification")
    .put("userId", 1001)
    .put("content", "New message")
    .put("timestamp", System.currentTimeMillis())
    .toString();

jedis.publish("notifications", message);
```

### 3. 异常处理

```java
JedisPubSub jedisPubSub = new JedisPubSub() {
    @Override
    public void onMessage(String channel, String message) {
        try {
            processMessage(channel, message);
        } catch (Exception e) {
            log.error("处理消息失败", e);
        }
    }
};
```

### 4. 优雅关闭

```java
// 取消订阅并关闭连接
Runtime.getRuntime().addShutdownHook(new Thread(() -> {
    jedisPubSub.unsubscribe();
    jedis.close();
}));
```

## 总结

- ✅ Pub/Sub 适合实时通知和广播场景
- ✅ 消息不持久化，适合可丢失的场景
- ✅ 支持模式匹配订阅
- ⚠️ 无消息确认和历史消息
- ⚠️ 订阅操作是阻塞的
- 💡 需要消息队列功能请使用 Stream

合理使用 Pub/Sub，能简化实时消息推送的实现！
