---
sidebar_position: 9
title: 场景设计题
---

# 🎯 场景设计题（专家级）

## 34. 如何设计一个延迟任务系统？

**答案要点：**

**方案对比：**

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 定时轮询 | 简单 | 精度低，性能差 | 小规模 |
| DelayQueue | 精度高 | 单机，不持久化 | 单机场景 |
| Redis ZSet | 分布式，持久化 | 需要轮询 | 中等规模 |
| 时间轮 | 高性能 | 实现复杂 | 高性能场景 |
| RocketMQ 延迟消息 | 可靠，分布式 | 延迟级别固定 | 大规模 |

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

## 35. 如何设计一个限流系统？

**答案要点：**

**限流算法对比：**

| 算法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 计数器 | 固定窗口计数 | 简单 | 临界问题 |
| 滑动窗口 | 滑动时间窗口 | 平滑 | 内存占用 |
| 漏桶 | 固定速率流出 | 平滑 | 无法应对突发 |
| 令牌桶 | 固定速率生成令牌 | 允许突发 | 实现复杂 |

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
