---
sidebar_position: 17
title: 面试题集
---

# Redis 面试题集

## 基础题

### 1. Redis 是什么？有什么优势？

**答：** Redis 是内存数据存储系统。

**优势：**

- 速度快（内存）
- 数据类型丰富
- 支持持久化
- 高可用
- 丰富的功能

### 2. Redis 为什么快？

**答：**

1. 内存操作
2. 单线程避免锁
3. I/O 多路复用
4. 高效数据结构

### 3. Redis 数据类型及使用场景

**答：**

| 类型       | 场景             |
| ---------- | ---------------- |
| String     | 缓存、计数器     |
| Hash       | 对象存储         |
| List       | 消息队列、时间线 |
| Set        | 去重、共同好友   |
| Sorted Set | 排行榜           |

### 4. 持久化方式及区别

**答：**

**RDB：**

- 快照
- 恢复快
- 可能丢数据

**AOF：**

- 追加日志
- 更安全
- 文件大

**建议：** 同时开启

### 5. 缓存穿透、击穿、雪崩

**答：**

**穿透：** 查询不存在数据 → 缓存空对象/布隆过滤器

**击穿：** 热 key 失效 → 互斥锁/永不过期

**雪崩：** 大量 key 失效 → 随机过期时间/多级缓存

## 中级题

### 6. 主从复制原理

**答：**

1. 从节点发送 PSYNC
2. 主节点判断全量/增量
3. 全量：RDB+缓冲区
4. 增量：发送差异命令
5. 持续同步

### 7. 哨兵工作原理

**答：**

1. 监控主从状态
2. 主观下线：单个哨兵判断
3. 客观下线：多数哨兵确认
4. 选举领导者
5. 故障转移

### 8. 集群如何分片

**答：**

- 16384 个槽位
- CRC16(key) % 16384
- 每个节点负责部分槽位
- 自动迁移槽位

### 9. 分布式锁实现

**答：**

```java
// SET NX EX
String result = jedis.set(
    lockKey,
    requestId,
    "NX",
    "EX",
    expireTime
);

// Lua 脚本释放
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
```

### 10. 如何保证缓存和数据库一致性

**答：**

**方案 1：** 延迟双删

```java
redis.del(key);
db.update(data);
Thread.sleep(500);
redis.del(key);
```

**方案 2：** 订阅 binlog 更新缓存

**方案 3：** 设置短过期时间

## 高级题

### 11. Redis 内存淘汰策略

**答：**

- **noeviction** - 不淘汰
- **allkeys-lru** - 全部 LRU
- **allkeys-lfu** - 全部 LFU
- **volatile-lru** - 过期 LRU
- **volatile-lfu** -过期 LFU
- **volatile-ttl** - 优先即将过期

### 12. Pipeline vs Transaction vs Lua

**答：**

| 特性     | Pipeline | Transaction | Lua    |
| -------- | -------- | ----------- | ------ |
| 原子性   | ❌       | ✅          | ✅     |
| 条件判断 | ❌       | ❌          | ✅     |
| 性能     | ⭐⭐⭐   | ⭐⭐        | ⭐⭐⭐ |

### 13. 热 key 问题

**答：**

**识别：**

```bash
redis-cli --hotkeys
```

**解决：**

1. 本地缓存
2. 读写分离
3. 分片降低单 key 压力

### 14. 大 key 问题

**答：**

**识别：**

```bash
redis-cli --bigkeys
```

**解决：**

1. 拆分 key
2. 压缩数据
3. 清理数据

### 15. Redis 单线程为何这么快

**答：**

1. **瓶颈不在 CPU** - 在内存和网络
2. **避免锁竞争**
3. **避免上下文切换**
4. **I/O 多路复用**

**注：** Redis 6.0 后引入 I/O 多线程

## 实战题

### 16. 设计排行榜系统

**答：**

```java
// 更新分数
zadd("leaderboard", score, playerId);

// 获取排名
zrevrank("leaderboard", playerId);

// 获取top 10
zrevrange("leaderboard", 0, 9, true);

// 获取我前后的人
long rank = zrevrank("leaderboard", myId);
zrevrange("leaderboard", rank - 5, rank + 5, true);
```

### 17. 设计限流器

**答：**

```lua
-- 滑动窗口限流
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

redis.call('ZADD', key, now, now)
redis.call('ZREMRANGEBYSCORE', key, 0, now - window * 1000)
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('EXPIRE', key, window)
    return 1
else
    return 0
end
```

### 18. 设计消息队列

**答：**

**List 方式：**

```bash
LPUSH queue msg
BRPOP queue 0
```

**Stream 方式：**

```bash
XADD stream * field value
XREADGROUP GROUP group consumer BLOCK 5000 STREAMS stream >
XACK stream group id
```

### 19. 设计分布式 Session

**答：**

```java
// 存储Session
String sessionId = UUID.randomUUID().toString();
Map<String, String> session = new HashMap<>();
session.put("userId", "1001");
session.put("username", "Alice");
redis.hmset("session:" + sessionId, session);
redis.expire("session:" + sessionId, 1800);

// 读取Session
Map<String, String> session = redis.hgetAll("session:" + sessionId);
```

### 20. 如何实现延迟队列

**答：**

```java
// 添加延迟任务
long executeTime = System.currentTimeMillis() + delay;
zadd("delay:queue", executeTime, taskId);

// 消费任务
while (true) {
    Set<String> tasks = zrangeByScore(
        "delay:queue",
        0,
        System.currentTimeMillis()
    );

    for (String task : tasks) {
        // 处理任务
        processTask(task);
        // 删除任务
        zrem("delay:queue", task);
    }

    Thread.sleep(1000);
}
```

## 场景设计题

### 21. 设计附近的人功能

**答：**

```java
// 添加位置
geoadd("locations", lng, lat, userId);

// 查询附近的人
List<GeoRadiusResponse> nearby = georadius(
    "locations",
    lng, lat,
    5, KILOMETERS
);
```

### 22. 设计计数器系统

**答：**

```java
// 简单计数
incr("page:views:" + pageId);

// 带限制的计数
String script =
    "local current = redis.call('INCR', KEYS[1]) " +
    "if current > tonumber(ARGV[1]) then " +
    "  redis.call('SET', KEYS[1], ARGV[1]) " +
    "  return ARGV[1] " +
    "end " +
    "return current";
```

### 23. 设计签到系统

**答：**

```bash
# 签到
SETBIT sign:202312:1001 10 1  # 12月10日签到

# 查询某天是否签到
GETBIT sign:202312:1001 10

# 统计签到天数
BITCOUNT sign:202312:1001
```

## 总结

**高频考点：**

- ✅ 数据类型和应用场景
- ✅ 持久化机制
- ✅ 缓存三大问题
- ✅ 主从复制/哨兵/集群
- ✅ 分布式锁
- ✅ 性能优化

**回答技巧：**

1. 先说原理
2. 再说优缺点
3. 最后说应用场景
4. 有代码示例更好

准备充分，面试必过！
