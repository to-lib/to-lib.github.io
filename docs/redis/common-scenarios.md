---
sidebar_position: 22
title: 常见业务场景
---

# Redis 常见业务场景实现

本文介绍 Redis 在实际项目中的常见使用场景和实现方案，包含完整的代码示例。

## 分布式 Session

### 问题背景

在微服务架构中，多个服务实例需要共享用户 Session 数据。

### 实现方案

使用 Spring Session + Redis：

```yaml
# application.yml
spring:
  session:
    store-type: redis
    timeout: 30m
  redis:
    host: localhost
    port: 6379
```

```java
@Configuration
@EnableRedisHttpSession(maxInactiveIntervalInSeconds = 1800)
public class SessionConfig {
    
    @Bean
    public RedisSerializer<Object> springSessionDefaultRedisSerializer() {
        return new GenericJackson2JsonRedisSerializer();
    }
}
```

### Session 操作

```java
@RestController
public class SessionController {
    
    @PostMapping("/login")
    public Result login(@RequestBody LoginRequest request, HttpSession session) {
        // 验证用户
        User user = userService.login(request.getUsername(), request.getPassword());
        if (user != null) {
            session.setAttribute("user", user);
            session.setAttribute("loginTime", System.currentTimeMillis());
            return Result.success("登录成功");
        }
        return Result.fail("用户名或密码错误");
    }
    
    @GetMapping("/user/info")
    public Result getUserInfo(HttpSession session) {
        User user = (User) session.getAttribute("user");
        if (user == null) {
            return Result.fail("未登录");
        }
        return Result.success(user);
    }
    
    @PostMapping("/logout")
    public Result logout(HttpSession session) {
        session.invalidate();
        return Result.success("退出成功");
    }
}
```

## 全局唯一 ID 生成器

### 方案一：INCR 自增

```java
@Component
public class RedisIdGenerator {
    
    @Autowired
    private StringRedisTemplate redisTemplate;
    
    private static final String ID_KEY = "id:generator:";
    
    /**
     * 生成唯一 ID
     * 格式: 时间戳(10位) + 序列号(5位)
     */
    public long nextId(String bizType) {
        // 获取当天日期作为 key 的一部分
        String today = LocalDate.now().format(DateTimeFormatter.BASIC_ISO_DATE);
        String key = ID_KEY + bizType + ":" + today;
        
        // 自增
        Long increment = redisTemplate.opsForValue().increment(key);
        
        // 设置过期时间（第二天过期）
        if (increment == 1) {
            redisTemplate.expire(key, Duration.ofDays(2));
        }
        
        // 组合 ID: 时间戳 + 序列号
        long timestamp = LocalDate.now().atStartOfDay()
            .toEpochSecond(ZoneOffset.of("+8"));
        return timestamp * 100000 + increment;
    }
}
```

### 方案二：雪花算法 + Redis 分配机器 ID

```java
@Component
public class SnowflakeIdGenerator {
    
    private static final long EPOCH = 1704067200000L; // 2024-01-01
    private static final long WORKER_ID_BITS = 10L;
    private static final long SEQUENCE_BITS = 12L;
    
    private final long workerId;
    private long sequence = 0L;
    private long lastTimestamp = -1L;
    
    @Autowired
    public SnowflakeIdGenerator(StringRedisTemplate redisTemplate) {
        // 从 Redis 获取机器 ID
        Long id = redisTemplate.opsForValue().increment("snowflake:worker:id");
        this.workerId = id % (1 << WORKER_ID_BITS);
    }
    
    public synchronized long nextId() {
        long timestamp = System.currentTimeMillis();
        
        if (timestamp < lastTimestamp) {
            throw new RuntimeException("时钟回拨");
        }
        
        if (timestamp == lastTimestamp) {
            sequence = (sequence + 1) & ((1 << SEQUENCE_BITS) - 1);
            if (sequence == 0) {
                timestamp = waitNextMillis(lastTimestamp);
            }
        } else {
            sequence = 0L;
        }
        
        lastTimestamp = timestamp;
        
        return ((timestamp - EPOCH) << (WORKER_ID_BITS + SEQUENCE_BITS))
                | (workerId << SEQUENCE_BITS)
                | sequence;
    }
    
    private long waitNextMillis(long lastTimestamp) {
        long timestamp = System.currentTimeMillis();
        while (timestamp <= lastTimestamp) {
            timestamp = System.currentTimeMillis();
        }
        return timestamp;
    }
}
```

## 延迟队列

### 基于 Sorted Set 实现

```java
@Component
public class DelayQueue {
    
    @Autowired
    private StringRedisTemplate redisTemplate;
    
    private static final String DELAY_QUEUE_KEY = "delay:queue:";
    
    /**
     * 添加延迟任务
     */
    public void addTask(String queueName, String taskId, long delaySeconds) {
        long executeTime = System.currentTimeMillis() + delaySeconds * 1000;
        redisTemplate.opsForZSet().add(
            DELAY_QUEUE_KEY + queueName,
            taskId,
            executeTime
        );
    }
    
    /**
     * 获取到期的任务
     */
    public Set<String> pollTasks(String queueName, int count) {
        long now = System.currentTimeMillis();
        String key = DELAY_QUEUE_KEY + queueName;
        
        // 获取到期的任务
        Set<String> tasks = redisTemplate.opsForZSet()
            .rangeByScore(key, 0, now, 0, count);
        
        if (tasks != null && !tasks.isEmpty()) {
            // 移除已获取的任务
            for (String task : tasks) {
                redisTemplate.opsForZSet().remove(key, task);
            }
        }
        
        return tasks;
    }
    
    /**
     * 取消任务
     */
    public void cancelTask(String queueName, String taskId) {
        redisTemplate.opsForZSet().remove(DELAY_QUEUE_KEY + queueName, taskId);
    }
}
```

### 消费者示例

```java
@Component
public class DelayQueueConsumer {
    
    @Autowired
    private DelayQueue delayQueue;
    
    @Scheduled(fixedDelay = 1000)  // 每秒执行
    public void consume() {
        Set<String> tasks = delayQueue.pollTasks("order:timeout", 10);
        if (tasks != null) {
            for (String taskId : tasks) {
                // 处理任务
                handleOrderTimeout(taskId);
            }
        }
    }
    
    private void handleOrderTimeout(String orderId) {
        // 订单超时处理逻辑
        log.info("处理超时订单: {}", orderId);
    }
}
```

### 使用示例

```java
// 订单创建后，30 分钟未支付自动取消
delayQueue.addTask("order:timeout", orderId, 30 * 60);

// 用户支付成功后，取消延迟任务
delayQueue.cancelTask("order:timeout", orderId);
```

## 滑动窗口限流

### 实现

```java
@Component
public class SlidingWindowRateLimiter {
    
    @Autowired
    private StringRedisTemplate redisTemplate;
    
    /**
     * 滑动窗口限流
     * @param key 限流 key
     * @param limit 限制次数
     * @param windowSeconds 窗口时间（秒）
     * @return true-允许, false-限流
     */
    public boolean isAllowed(String key, int limit, int windowSeconds) {
        long now = System.currentTimeMillis();
        long windowStart = now - windowSeconds * 1000L;
        
        String rateLimitKey = "ratelimit:" + key;
        
        // 使用 Lua 脚本保证原子性
        String script = """
            -- 移除窗口外的请求
            redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[1])
            
            -- 获取当前窗口内的请求数
            local count = redis.call('ZCARD', KEYS[1])
            
            if count < tonumber(ARGV[2]) then
                -- 未超过限制，添加当前请求
                redis.call('ZADD', KEYS[1], ARGV[3], ARGV[3])
                redis.call('EXPIRE', KEYS[1], ARGV[4])
                return 1
            else
                return 0
            end
            """;
        
        Long result = redisTemplate.execute(
            new DefaultRedisScript<>(script, Long.class),
            Collections.singletonList(rateLimitKey),
            String.valueOf(windowStart),
            String.valueOf(limit),
            String.valueOf(now),
            String.valueOf(windowSeconds)
        );
        
        return Long.valueOf(1).equals(result);
    }
}
```

### 使用示例

```java
@RestController
public class ApiController {
    
    @Autowired
    private SlidingWindowRateLimiter rateLimiter;
    
    @GetMapping("/api/data")
    public Result getData(HttpServletRequest request) {
        String clientIp = request.getRemoteAddr();
        
        // 每个 IP 每分钟最多 100 次请求
        if (!rateLimiter.isAllowed(clientIp, 100, 60)) {
            return Result.fail("请求过于频繁，请稍后再试");
        }
        
        // 正常处理
        return Result.success(dataService.getData());
    }
}
```

### AOP 注解方式

```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface RateLimit {
    int limit() default 100;
    int window() default 60;
    String key() default "";
}

@Aspect
@Component
public class RateLimitAspect {
    
    @Autowired
    private SlidingWindowRateLimiter rateLimiter;
    
    @Around("@annotation(rateLimit)")
    public Object around(ProceedingJoinPoint point, RateLimit rateLimit) throws Throwable {
        String key = rateLimit.key();
        if (key.isEmpty()) {
            key = point.getSignature().toShortString();
        }
        
        if (!rateLimiter.isAllowed(key, rateLimit.limit(), rateLimit.window())) {
            throw new RuntimeException("请求过于频繁");
        }
        
        return point.proceed();
    }
}

// 使用
@RateLimit(limit = 10, window = 60, key = "api:sendSms")
public void sendSms(String phone) {
    // ...
}
```

## 排行榜

### 实现

```java
@Service
public class LeaderboardService {
    
    @Autowired
    private StringRedisTemplate redisTemplate;
    
    private static final String LEADERBOARD_KEY = "leaderboard:";
    
    /**
     * 更新分数
     */
    public void updateScore(String boardName, String userId, double score) {
        redisTemplate.opsForZSet().add(LEADERBOARD_KEY + boardName, userId, score);
    }
    
    /**
     * 增加分数
     */
    public Double incrementScore(String boardName, String userId, double delta) {
        return redisTemplate.opsForZSet()
            .incrementScore(LEADERBOARD_KEY + boardName, userId, delta);
    }
    
    /**
     * 获取用户排名（从 0 开始）
     */
    public Long getRank(String boardName, String userId) {
        return redisTemplate.opsForZSet()
            .reverseRank(LEADERBOARD_KEY + boardName, userId);
    }
    
    /**
     * 获取用户分数
     */
    public Double getScore(String boardName, String userId) {
        return redisTemplate.opsForZSet()
            .score(LEADERBOARD_KEY + boardName, userId);
    }
    
    /**
     * 获取排行榜（Top N）
     */
    public List<UserRank> getTopN(String boardName, int n) {
        Set<ZSetOperations.TypedTuple<String>> tuples = redisTemplate.opsForZSet()
            .reverseRangeWithScores(LEADERBOARD_KEY + boardName, 0, n - 1);
        
        List<UserRank> result = new ArrayList<>();
        if (tuples != null) {
            int rank = 1;
            for (ZSetOperations.TypedTuple<String> tuple : tuples) {
                result.add(new UserRank(rank++, tuple.getValue(), tuple.getScore()));
            }
        }
        return result;
    }
    
    /**
     * 获取用户周围的排名
     */
    public List<UserRank> getAroundRank(String boardName, String userId, int range) {
        Long rank = getRank(boardName, userId);
        if (rank == null) {
            return Collections.emptyList();
        }
        
        long start = Math.max(0, rank - range);
        long end = rank + range;
        
        Set<ZSetOperations.TypedTuple<String>> tuples = redisTemplate.opsForZSet()
            .reverseRangeWithScores(LEADERBOARD_KEY + boardName, start, end);
        
        List<UserRank> result = new ArrayList<>();
        if (tuples != null) {
            int currentRank = (int) start + 1;
            for (ZSetOperations.TypedTuple<String> tuple : tuples) {
                result.add(new UserRank(currentRank++, tuple.getValue(), tuple.getScore()));
            }
        }
        return result;
    }
}

@Data
@AllArgsConstructor
public class UserRank {
    private int rank;
    private String userId;
    private Double score;
}
```

### 使用示例

```java
// 游戏结束后更新分数
leaderboardService.incrementScore("game:daily", "user:1001", 100);

// 获取排行榜 Top 10
List<UserRank> top10 = leaderboardService.getTopN("game:daily", 10);

// 获取用户排名
Long rank = leaderboardService.getRank("game:daily", "user:1001");

// 获取用户周围的排名
List<UserRank> around = leaderboardService.getAroundRank("game:daily", "user:1001", 5);
```

## 抢红包

### 实现

```java
@Service
public class RedPacketService {
    
    @Autowired
    private StringRedisTemplate redisTemplate;
    
    private static final String RED_PACKET_KEY = "redpacket:";
    private static final String RED_PACKET_RECORD = "redpacket:record:";
    
    /**
     * 创建红包
     */
    public void createRedPacket(String packetId, double totalAmount, int count) {
        // 使用二倍均值法分配红包
        List<Double> amounts = divideRedPacket(totalAmount, count);
        
        // 存入 Redis List
        String key = RED_PACKET_KEY + packetId;
        for (Double amount : amounts) {
            redisTemplate.opsForList().rightPush(key, String.format("%.2f", amount));
        }
        
        // 设置过期时间（24 小时）
        redisTemplate.expire(key, Duration.ofHours(24));
    }
    
    /**
     * 抢红包
     */
    public Double grabRedPacket(String packetId, String userId) {
        String recordKey = RED_PACKET_RECORD + packetId;
        
        // 检查是否已抢过
        if (redisTemplate.opsForSet().isMember(recordKey, userId)) {
            throw new RuntimeException("您已经抢过这个红包了");
        }
        
        // 抢红包（原子操作）
        String key = RED_PACKET_KEY + packetId;
        String amount = redisTemplate.opsForList().leftPop(key);
        
        if (amount == null) {
            throw new RuntimeException("红包已被抢完");
        }
        
        // 记录抢红包
        redisTemplate.opsForSet().add(recordKey, userId);
        redisTemplate.expire(recordKey, Duration.ofHours(24));
        
        return Double.parseDouble(amount);
    }
    
    /**
     * 二倍均值法分配红包
     */
    private List<Double> divideRedPacket(double totalAmount, int count) {
        List<Double> amounts = new ArrayList<>();
        Random random = new Random();
        double remaining = totalAmount;
        
        for (int i = 0; i < count - 1; i++) {
            // 最小 0.01，最大是剩余平均值的 2 倍
            double max = remaining / (count - i) * 2;
            double amount = Math.max(0.01, random.nextDouble() * max);
            amount = Math.round(amount * 100) / 100.0;
            amounts.add(amount);
            remaining -= amount;
        }
        
        // 最后一个红包
        amounts.add(Math.round(remaining * 100) / 100.0);
        return amounts;
    }
}
```

## 购物车

### 实现

```java
@Service
public class CartService {
    
    @Autowired
    private StringRedisTemplate redisTemplate;
    
    private static final String CART_KEY = "cart:";
    
    /**
     * 添加商品到购物车
     */
    public void addItem(String userId, String productId, int quantity) {
        String key = CART_KEY + userId;
        
        // 如果商品已存在，增加数量
        Object existing = redisTemplate.opsForHash().get(key, productId);
        if (existing != null) {
            quantity += Integer.parseInt(existing.toString());
        }
        
        redisTemplate.opsForHash().put(key, productId, String.valueOf(quantity));
    }
    
    /**
     * 修改商品数量
     */
    public void updateQuantity(String userId, String productId, int quantity) {
        String key = CART_KEY + userId;
        if (quantity <= 0) {
            redisTemplate.opsForHash().delete(key, productId);
        } else {
            redisTemplate.opsForHash().put(key, productId, String.valueOf(quantity));
        }
    }
    
    /**
     * 移除商品
     */
    public void removeItem(String userId, String productId) {
        redisTemplate.opsForHash().delete(CART_KEY + userId, productId);
    }
    
    /**
     * 获取购物车
     */
    public Map<String, Integer> getCart(String userId) {
        Map<Object, Object> entries = redisTemplate.opsForHash()
            .entries(CART_KEY + userId);
        
        Map<String, Integer> cart = new HashMap<>();
        entries.forEach((k, v) -> cart.put(k.toString(), Integer.parseInt(v.toString())));
        return cart;
    }
    
    /**
     * 清空购物车
     */
    public void clearCart(String userId) {
        redisTemplate.delete(CART_KEY + userId);
    }
    
    /**
     * 获取购物车商品数量
     */
    public Long getItemCount(String userId) {
        return redisTemplate.opsForHash().size(CART_KEY + userId);
    }
}
```

## 点赞功能

### 实现

```java
@Service
public class LikeService {
    
    @Autowired
    private StringRedisTemplate redisTemplate;
    
    private static final String LIKE_KEY = "like:";
    private static final String LIKE_COUNT_KEY = "like:count:";
    
    /**
     * 点赞
     */
    public boolean like(String targetId, String userId) {
        String key = LIKE_KEY + targetId;
        Long added = redisTemplate.opsForSet().add(key, userId);
        
        if (added != null && added > 0) {
            // 增加点赞计数
            redisTemplate.opsForValue().increment(LIKE_COUNT_KEY + targetId);
            return true;
        }
        return false;
    }
    
    /**
     * 取消点赞
     */
    public boolean unlike(String targetId, String userId) {
        String key = LIKE_KEY + targetId;
        Long removed = redisTemplate.opsForSet().remove(key, userId);
        
        if (removed != null && removed > 0) {
            redisTemplate.opsForValue().decrement(LIKE_COUNT_KEY + targetId);
            return true;
        }
        return false;
    }
    
    /**
     * 是否已点赞
     */
    public boolean isLiked(String targetId, String userId) {
        return Boolean.TRUE.equals(
            redisTemplate.opsForSet().isMember(LIKE_KEY + targetId, userId)
        );
    }
    
    /**
     * 获取点赞数
     */
    public long getLikeCount(String targetId) {
        String count = redisTemplate.opsForValue().get(LIKE_COUNT_KEY + targetId);
        return count == null ? 0 : Long.parseLong(count);
    }
    
    /**
     * 获取点赞用户列表
     */
    public Set<String> getLikeUsers(String targetId) {
        return redisTemplate.opsForSet().members(LIKE_KEY + targetId);
    }
}
```

## 在线用户统计

### 使用 Bitmap 实现

```java
@Service
public class OnlineUserService {
    
    @Autowired
    private StringRedisTemplate redisTemplate;
    
    private static final String ONLINE_KEY = "online:users:";
    
    /**
     * 用户上线
     */
    public void online(String date, long userId) {
        redisTemplate.opsForValue().setBit(ONLINE_KEY + date, userId, true);
    }
    
    /**
     * 用户下线
     */
    public void offline(String date, long userId) {
        redisTemplate.opsForValue().setBit(ONLINE_KEY + date, userId, false);
    }
    
    /**
     * 检查用户是否在线
     */
    public boolean isOnline(String date, long userId) {
        return Boolean.TRUE.equals(
            redisTemplate.opsForValue().getBit(ONLINE_KEY + date, userId)
        );
    }
    
    /**
     * 统计在线用户数
     */
    public long countOnlineUsers(String date) {
        // 使用 BITCOUNT 命令
        return redisTemplate.execute((RedisCallback<Long>) connection -> 
            connection.bitCount(
                (ONLINE_KEY + date).getBytes()
            )
        );
    }
    
    /**
     * 统计连续登录用户
     */
    public long countContinuousUsers(String... dates) {
        // 计算多天的 AND 结果
        byte[][] keys = Arrays.stream(dates)
            .map(d -> (ONLINE_KEY + d).getBytes())
            .toArray(byte[][]::new);
        
        String resultKey = "online:continuous";
        redisTemplate.execute((RedisCallback<Long>) connection -> {
            connection.bitOp(RedisStringCommands.BitOperation.AND, 
                resultKey.getBytes(), keys);
            return null;
        });
        
        return redisTemplate.execute((RedisCallback<Long>) connection ->
            connection.bitCount(resultKey.getBytes())
        );
    }
}
```

## 小结

| 场景 | Redis 数据结构 | 核心命令 |
|------|---------------|----------|
| 分布式 Session | Hash | HSET, HGETALL |
| 唯一 ID | String | INCR |
| 延迟队列 | Sorted Set | ZADD, ZRANGEBYSCORE |
| 滑动窗口限流 | Sorted Set | ZADD, ZREMRANGEBYSCORE, ZCARD |
| 排行榜 | Sorted Set | ZADD, ZREVRANGE, ZRANK |
| 抢红包 | List | LPOP |
| 购物车 | Hash | HSET, HGETALL |
| 点赞 | Set | SADD, SISMEMBER |
| 在线统计 | Bitmap | SETBIT, BITCOUNT |

**最佳实践**：

- ✅ 选择合适的数据结构
- ✅ 使用 Lua 脚本保证原子性
- ✅ 设置合理的过期时间
- ✅ 考虑异常处理和边界情况
