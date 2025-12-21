---
sidebar_position: 9
title: 场景设计题
---

# 🎯 场景设计题（专家级）

## 37. 如何设计一个延迟任务系统？

**答案要点：**

**方案对比：**

| 方案              | 优点           | 缺点           | 适用场景   |
| ----------------- | -------------- | -------------- | ---------- |
| 定时轮询          | 简单           | 精度低，性能差 | 小规模     |
| DelayQueue        | 精度高         | 单机，不持久化 | 单机场景   |
| Redis ZSet        | 分布式，持久化 | 需要轮询       | 中等规模   |
| 时间轮            | 高性能         | 实现复杂       | 高性能场景 |
| RocketMQ 延迟消息 | 可靠，分布式   | 延迟级别固定   | 大规模     |

**Redis ZSet 实现：**

```java
@Service
public class DelayTaskService {
    @Autowired
    private StringRedisTemplate redisTemplate;

    private static final String DELAY_QUEUE = "delay:queue";

    // 添加延迟任务
    public void addTask(String taskId, long delaySeconds) {
        long executeTime = System.currentTimeMillis() + delaySeconds * 1000;
        redisTemplate.opsForZSet().add(DELAY_QUEUE, taskId, executeTime);
    }

    // 消费延迟任务
    @Scheduled(fixedRate = 1000)
    public void consumeTasks() {
        long now = System.currentTimeMillis();
        Set<String> tasks = redisTemplate.opsForZSet()
            .rangeByScore(DELAY_QUEUE, 0, now);

        for (String taskId : tasks) {
            // 原子性移除并处理
            Long removed = redisTemplate.opsForZSet().remove(DELAY_QUEUE, taskId);
            if (removed != null && removed > 0) {
                processTask(taskId);
            }
        }
    }
}
```

**时间轮算法原理：**

```
时间轮（类似钟表）
     0
   7   1
  6     2
   5   3
     4

- 每个槽位存储该时刻到期的任务
- 指针每隔固定时间移动一格
- 支持多层时间轮处理长延迟
```

**延伸：** 参考 [消息队列](/docs/rocketmq)

---

## 38. 如何设计一个限流系统？

**答案要点：**

**限流算法对比：**

| 算法     | 原理             | 优点     | 缺点         |
| -------- | ---------------- | -------- | ------------ |
| 计数器   | 固定窗口计数     | 简单     | 临界问题     |
| 滑动窗口 | 滑动时间窗口     | 平滑     | 内存占用     |
| 漏桶     | 固定速率流出     | 平滑     | 无法应对突发 |
| 令牌桶   | 固定速率生成令牌 | 允许突发 | 实现复杂     |

**令牌桶算法实现：**

```java
public class TokenBucketRateLimiter {
    private final long capacity;        // 桶容量
    private final long refillRate;      // 每秒填充令牌数
    private long tokens;                // 当前令牌数
    private long lastRefillTime;        // 上次填充时间

    public TokenBucketRateLimiter(long capacity, long refillRate) {
        this.capacity = capacity;
        this.refillRate = refillRate;
        this.tokens = capacity;
        this.lastRefillTime = System.currentTimeMillis();
    }

    public synchronized boolean tryAcquire() {
        refill();
        if (tokens > 0) {
            tokens--;
            return true;
        }
        return false;
    }

    private void refill() {
        long now = System.currentTimeMillis();
        long elapsed = now - lastRefillTime;
        long tokensToAdd = elapsed * refillRate / 1000;
        tokens = Math.min(capacity, tokens + tokensToAdd);
        lastRefillTime = now;
    }
}
```

**Redis + Lua 分布式限流：**

```java
public class RedisRateLimiter {
    private static final String SCRIPT =
        "local key = KEYS[1] " +
        "local limit = tonumber(ARGV[1]) " +
        "local window = tonumber(ARGV[2]) " +
        "local current = tonumber(redis.call('get', key) or '0') " +
        "if current + 1 > limit then " +
        "   return 0 " +
        "else " +
        "   redis.call('incrby', key, 1) " +
        "   redis.call('expire', key, window) " +
        "   return 1 " +
        "end";

    public boolean tryAcquire(String key, int limit, int windowSeconds) {
        Long result = redisTemplate.execute(
            new DefaultRedisScript<>(SCRIPT, Long.class),
            Collections.singletonList(key),
            String.valueOf(limit),
            String.valueOf(windowSeconds)
        );
        return result != null && result == 1;
    }
}
```

**延伸：** 参考 [微服务 - 服务治理](/docs/microservices/service-governance)

---

## 39. 如何设计一个短链接系统（URL Shortener）？

**答案要点：**

**核心原理：** 将长 URL 映射为短字符串（如 `Bit.ly/3h7f9`）。

**ID 生成策略：**

- **数据库自增 ID**：简单，但有单点瓶颈，ID 非随机。
- **Redis INCR**：性能好，需处理持久化。
- **Snowflake 算法**：分布式 ID，高性能，ID 较长。
- **MurmurHash**：哈希算法，可能冲突。

**推荐方案：分布式 ID + Base62 编码**

1.  **ID 生成**：使用 Snowflake 或 Redis 生成唯一 ID（如 10000000001）。
2.  **Base62 编码**：将 10 进制 ID 转为 62 进制（0-9, a-z, A-Z）。
    - `10000000001` -> `aB3dE`
3.  **存储**：Redis（热数据） + MySQL（冷数据）。
    - Key: ShortURL, Value: LongURL

**重定向流程：**

1.  用户访问 `http://short.url/aB3dE`。
2.  服务查询 Redis/DB 获取长 URL。
3.  服务返回 HTTP 302（临时重定向）或 301（永久重定向）到长 URL。
    - **301**：浏览器缓存，服务器压力小，但无法统计点击量。
    - **302**：每次通过服务器，方便统计，服务器压力大。

---

## 40. 如何设计一个实时排行榜系统？

**答案要点：**

**技术选型：Redis Sorted Set (ZSet)**

- **ZADD key score member**：添加/更新排名（时间复杂度 O(logN)）。
- **ZREVRANGE key start end**：获取 Top N 用户。
- **ZRANK/ZREVRANK key member**：获取特定用户排名。

**百万级用户排行榜优化：**

1.  **分桶策略**：如果只是 Top 100，不需要全量排序。可以将用户按积分范围分桶，只对高分桶进行排序。
2.  **主要 ID 映射**：ZSet 存储 `UserId`，详细信息从 User 表查（或缓存）。

**如果数据量超大（千万/亿级）：**

- **Redis 集群**：按 Key 分片（如 `leaderboard:daily`, `leaderboard:weekly`）。
- **离线计算**：使用 Spark/Flink 计算全量排名，Redis 只存 Top 1000。
- **概率算法**：使用 Count-Min Sketch 估算（针对不需要精确排名的场景）。

---

## 41. 如何设计一个分布式日志收集系统？

**答案要点：**

**业界标准方案：ELK Stack (Elasticsearch, Logstash, Kibana)**

**架构流程：**

```
应用服务器 (App)
   |
   v (Filebeat/Logstash Agent)
Kafka (消息队列/缓冲)
   |
   v (Logstash/Fluentd 消费)
Elasticsearch (索引与存储)
   |
   v
Kibana (可视化查询)
```

**关键设计点：**

1.  **Agent（采集层）**：轻量级，部署在业务机器，读取日志文件（Tail）发送到 MQ。
2.  **Buffer（缓冲层）**：Kafka，削峰填谷，防止 ES 在高并发下写入崩溃。
3.  **Parsing（处理层）**：Logstash/Fluentd，解析日志（正则、JSON），脱敏，格式化。
4.  **Storage（存储层）**：Elasticsearch，倒排索引，支持全文检索。按天分索引（`log-2023.10.01`）。
5.  **定期清理**：Curator 工具定期删除旧索引。
6.  **Trace ID**：全链路追踪（SkyWalking/Zipkin），在日志中注入 Trace ID，串联调用链。

**延伸：** 参考 [微服务 - 可观测性](/docs/microservices/observability)

---
