---
sidebar_position: 23
title: 实战案例
---

# Redis 实战案例

## 分布式锁

### 基础实现

```java
public class RedisLock {
    private Jedis jedis;
    private String lockKey;
    private String requestId;

    public boolean tryLock(int expireTime) {
        requestId = UUID.randomUUID().toString();
        String result = jedis.set(
            lockKey,
            requestId,
            "NX",
            "EX",
            expireTime
        );
        return "OK".equals(result);
    }

    public void unlock() {
        String script =
            "if redis.call('GET', KEYS[1]) == ARGV[1] then " +
            "  return redis.call('DEL', KEYS[1]) " +
            "else " +
            "  return 0 " +
            "end";

        jedis.eval(script,
            Collections.singletonList(lockKey),
            Collections.singletonList(requestId)
        );
    }
}
```

### Redisson 实现

```java
@Autowired
private RedissonClient redissonClient;

public void demo() {
    RLock lock = redissonClient.getLock("myLock");

    try {
        // 尝试加锁，最多等待10秒，锁30秒后自动释放
        if (lock.tryLock(10, 30, TimeUnit.SECONDS)) {
            try {
                // 业务逻辑
                doSomething();
            } finally {
                lock.unlock();
            }
        }
    } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
    }
}
```

## 限流器

### 固定窗口限流

```java
public boolean isAllowed(String userId, int limit, int window) {
    String key = "rate:limit:" + userId;
    Long current = jedis.incr(key);

    if (current == 1) {
        jedis.expire(key, window);
    }

    return current <= limit;
}
```

### 滑动窗口限流

```java
public boolean isAllowed(String userId, int limit, int window) {
    String key = "rate:limit:" + userId;
    long now = System.currentTimeMillis();

    // 删除window之外的记录
    jedis.zremrangeByScore(key, 0, now - window * 1000);

    // 统计window内的请求数
    long count = jedis.zcard(key);

    if (count < limit) {
        jedis.zadd(key, now, String.valueOf(now));
        jedis.expire(key, window + 1);
        return true;
    }

    return false;
}
```

### 令牌桶限流

```lua
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local tokens_key = key .. ':tokens'
local timestamp_key = key .. ':timestamp'

local last_tokens = tonumber(redis.call('GET', tokens_key)) or capacity
local last_time = tonumber(redis.call('GET', timestamp_key)) or now

local delta = math.max(0, now - last_time)
local filled_tokens = math.min(capacity, last_tokens + (delta * rate))

if filled_tokens >= requested then
    local new_tokens = filled_tokens - requested
    redis.call('SET', tokens_key, new_tokens)
    redis.call('SET', timestamp_key, now)
    return 1
else
    return 0
end
```

## 排行榜系统

### 基本实现

```java
public class Leaderboard {
    private Jedis jedis;
    private String key = "leaderboard";

    // 更新分数
    public void updateScore(String playerId, double score) {
        jedis.zadd(key, score, playerId);
    }

    // 获取排名（从1开始）
    public long getRank(String playerId) {
        Long rank = jedis.zrevrank(key, playerId);
        return rank == null ? -1 : rank + 1;
    }

    // 获取分数
    public Double getScore(String playerId) {
        return jedis.zscore(key, playerId);
    }

    // 获取Top N
    public List<Tuple> getTopN(int n) {
        return jedis.zrevrangeWithScores(key, 0, n - 1);
    }

    // 获取排名范围
    public List<Tuple> getRangeByRank(long start, long end) {
        return jedis.zrevrangeWithScores(key, start - 1, end - 1);
    }

    // 获取我前后的排名
    public List<Tuple> getAroundMe(String playerId, int count) {
        Long rank = jedis.zrevrank(key, playerId);
        if (rank == null) return Collections.emptyList();

        long start = Math.max(0, rank - count);
        long end = rank + count;

        return jedis.zrevrangeWithScores(key, start, end);
    }
}
```

## 社交关系

### 关注/粉丝

```java
public class SocialGraph {
    private Jedis jedis;

    // 关注
    public void follow(String userId, String targetId) {
        // userId 关注了 targetId
        jedis.sadd("following:" + userId, targetId);
        // targetId 的粉丝增加 userId
        jedis.sadd("followers:" + targetId, userId);
    }

    // 取消关注
    public void unfollow(String userId, String targetId) {
        jedis.srem("following:" + userId, targetId);
        jedis.srem("followers:" + targetId, userId);
    }

    // 共同关注
    public Set<String> commonFollowing(String userId1, String userId2) {
        return jedis.sinter(
            "following:" + userId1,
            "following:" + userId2
        );
    }

    // 是否互相关注
    public boolean isMutualFollow(String userId1, String userId2) {
        return jedis.sismember("following:" + userId1, userId2)
            && jedis.sismember("following:" + userId2, userId1);
    }
}
```

### 朋友圈

```java
public class Timeline {
    private Jedis jedis;

    // 发布动态
    public void publish(String userId, String content) {
        long postId = jedis.incr("post:id");

        // 保存动态内容
        jedis.hmset("post:" + postId, Map.of(
            "userId", userId,
            "content", content,
            "timestamp", String.valueOf(System.currentTimeMillis())
        ));

        // 推送到所有粉丝的时间线
        Set<String> followers = jedis.smembers("followers:" + userId);
        for (String follower : followers) {
            jedis.zadd("timeline:" + follower, postId, String.valueOf(postId));
        }

        // 推送到自己的时间线
        jedis.zadd("timeline:" + userId, postId, String.valueOf(postId));
    }

    // 获取时间线
    public List<Map<String, String>> getTimeline(String userId, int count) {
        Set<String> postIds = jedis.zrevrange("timeline:" + userId, 0, count - 1);

        List<Map<String, String>> posts = new ArrayList<>();
        for (String postId : postIds) {
            Map<String, String> post = jedis.hgetAll("post:" + postId);
            posts.add(post);
        }

        return posts;
    }
}
```

## 消息队列

### List 实现

```java
// 生产者
jedis.lpush("queue:tasks", task);

// 消费者
while (true) {
    List<String> result = jedis.brpop(5, "queue:tasks");
    if (result != null) {
        String task = result.get(1);
        processTask(task);
    }
}
```

### Stream 实现

```java
// 生产者
jedis.xadd("stream:tasks", "*", Map.of(
    "type", "email",
    "to", "user@example.com",
    "subject", "Welcome"
));

// 消费组
jedis.xgroupCreate("stream:tasks", "workers", "0", true);

// 消费者
while (true) {
    List<Map.Entry<String, List<StreamEntry>>> result =
        jedis.xreadGroup(
            "workers",
            "worker-1",
            XReadGroupParams.xReadGroupParams().count(10).block(5000),
            Map.of("stream:tasks", ">")
        );

    if (result != null && !result.isEmpty()) {
        for (Map.Entry<String, List<StreamEntry>> entry : result) {
            for (StreamEntry msg : entry.getValue()) {
                processMessage(msg);
                jedis.xack("stream:tasks", "workers", msg.getID());
            }
        }
    }
}
```

## 库存扣减

### 基础实现

```java
public boolean deductStock(String productId, int quantity) {
    String key = "stock:" + productId;

    String script =
        "local stock = redis.call('GET', KEYS[1]) " +
        "if not stock or tonumber(stock) < tonumber(ARGV[1]) then " +
        "  return 0 " +
        "end " +
        "redis.call('DECRBY', KEYS[1], ARGV[1]) " +
        "return 1";

    Long result = (Long) jedis.eval(
        script,
        Collections.singletonList(key),
        Collections.singletonList(String.valueOf(quantity))
    );

    return result == 1;
}
```

### 预减库存

```java
// 预减库存
public boolean preDeductStock(String orderId, String productId, int quantity) {
    String stockKey = "stock:" + productId;
    String orderKey = "order:" + orderId;

    String script =
        "local stock = redis.call('GET', KEYS[1]) " +
        "if not stock or tonumber(stock) < tonumber(ARGV[1]) then " +
        "  return 0 " +
        "end " +
        "redis.call('DECRBY', KEYS[1], ARGV[1]) " +
        "redis.call('SET', KEYS[2], ARGV[1]) " +
        "redis.call('EXPIRE', KEYS[2], 600) " +  // 10分钟过期
        "return 1";

    return (Long) jedis.eval(script,
        Arrays.asList(stockKey, orderKey),
        Collections.singletonList(String.valueOf(quantity))
    ) == 1;
}

// 恢复库存（订单取消）
public void restoreStock(String orderId, String productId) {
    String orderKey = "order:" + orderId;
    String stockKey = "stock:" + productId;

    String quantity = jedis.get(orderKey);
    if (quantity != null) {
        jedis.incrby(stockKey, Long.parseLong(quantity));
        jedis.del(orderKey);
    }
}
```

## 签到系统

```java
public class CheckIn {
    private Jedis jedis;

    // 签到
    public boolean checkIn(String userId, int year, int month, int day) {
        String key = String.format("checkin:%d%02d:%s", year, month, userId);
        jedis.setbit(key, day - 1, true);
        jedis.expire(key, 60 * 60 * 24 * 31);  // 31天过期
        return true;
    }

    // 查询某天是否签到
    public boolean isCheckedIn(String userId, int year, int month, int day) {
        String key = String.format("checkin:%d%02d:%s", year, month, userId);
        return jedis.getbit(key, day - 1);
    }

    // 统计签到天数
    public long countCheckIn(String userId, int year, int month) {
        String key = String.format("checkin:%d%02d:%s", year, month, userId);
        return jedis.bitcount(key);
    }

    // 获取连续签到天数
    public int getContinuousDays(String userId, int year, int month, int today) {
        String key = String.format("checkin:%d%02d:%s", year, month, userId);
        int continuous = 0;

        for (int day = today; day >= 1; day--) {
            if (jedis.getbit(key, day - 1)) {
                continuous++;
            } else {
                break;
            }
        }

        return continuous;
    }
}
```

## 总结

这些实战案例涵盖了：

- ✅ 分布式锁
- ✅ 限流器
- ✅ 排行榜
- ✅ 社交关系
- ✅ 消息队列
- ✅ 库存扣减
- ✅ 签到系统

结合实际场景，灵活运用 Redis！
