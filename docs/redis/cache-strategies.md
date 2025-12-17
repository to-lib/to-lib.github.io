---
sidebar_position: 17
title: Redis 缓存策略
---

# Redis 缓存策略

Redis 作为缓存的常见问题和解决方案。

## 常见缓存问题

### 1. 缓存穿透

查询一个**不存在的数据**，缓存和数据库都没有，导致每次请求都落到数据库。

#### 问题示例

```java
// 查询不存在的用户 ID: 99999
User user = redis.get("user:99999");
if (user == null) {
    user = database.query("SELECT * FROM users WHERE id = 99999");
    // user 为 null，不会缓存
    // 每次请求都会查询数据库
}
```

#### 解决方案

**方案一：缓存空值**

```java
User user = redis.get("user:99999");
if (user == null) {
    user = database.query("SELECT * FROM users WHERE id = 99999");
    if (user == null) {
        // 缓存空值，设置较短的过期时间
        redis.setex("user:99999", 60, "null");
    } else {
        redis.setex("user:99999", 3600, user);
    }
}
```

**方案二：布隆过滤器**

```java
// 启动时加载所有存在的用户 ID 到布隆过滤器
BloomFilter<Integer> bloomFilter = BloomFilter.create(...);
for (int userId : database.getAllUserIds()) {
    bloomFilter.put(userId);
}

// 查询时先判断
if (!bloomFilter.mightContain(99999)) {
    return null;  // 一定不存在
}

// 可能存在，查询缓存和数据库
User user = redis.get("user:99999");
if (user == null) {
    user = database.query(...);
}
```

### 2. 缓存击穿

一个**热点数据**过期，大量并发请求同时查询数据库。

#### 问题示例

```java
// 热点商品缓存过期，大量请求同时查询数据库
Product product = redis.get("product:1001");
if (product == null) {
    // 100 个并发请求同时执行这里
    product = database.query(...);
    redis.setex("product:1001", 3600, product);
}
```

#### 解决方案

**方案一：互斥锁**

```java
Product product = redis.get("product:1001");
if (product == null) {
    // 获取分布式锁
    String lockKey = "lock:product:1001";
    if (redis.setnx(lockKey, "1", 10)) {  // 10 秒超时
        try {
            // 双重检查
            product = redis.get("product:1001");
            if (product == null) {
                product = database.query(...);
                redis.setex("product:1001", 3600, product);
            }
        } finally {
            redis.del(lockKey);
        }
    } else {
        // 等待一段时间后重试
        Thread.sleep(100);
        return getProduct(1001);  // 递归重试
    }
}
```

**方案二：热点数据永不过期**

```java
// 逻辑过期：在缓存值中存储过期时间
class CacheData {
    Object data;
    long expireTime;
}

CacheData cacheData = redis.get("product:1001");
if (cacheData == null) {
    // 首次查询，设置数据（永不过期）
    Product product = database.query(...);
    cacheData = new CacheData(product, System.currentTimeMillis() + 3600000);
    redis.set("product:1001", cacheData);  // 不设置过期时间
} else if (System.currentTimeMillis() > cacheData.expireTime) {
    // 过期了，异步更新
    threadPool.submit(() -> {
        Product product = database.query(...);
        cacheData = new CacheData(product, System.currentTimeMillis() + 3600000);
        redis.set("product:1001", cacheData);
    });
    // 返回旧数据
}
return cacheData.data;
```

### 3. 缓存雪崩

大量缓存**同时过期**，导致请求全部落到数据库。

#### 问题示例

```java
// 批量缓存商品，设置相同的过期时间
for (Product product : products) {
    redis.setex("product:" + product.getId(), 3600, product);
}
// 1 小时后，所有缓存同时过期，数据库压力激增
```

#### 解决方案

**方案一：过期时间加随机值**

```java
// 过期时间加上随机值，避免同时过期
int baseExpire = 3600;
int randomExpire = new Random().nextInt(300);  // 0-300 秒
redis.setex("product:" + productId, baseExpire + randomExpire, product);
```

**方案二：多级缓存**

```
Client -> Local Cache (5m) -> Redis (1h) -> Database
```

**方案三：限流降级**

```java
// 使用限流器保护数据库
RateLimiter rateLimiter = RateLimiter.create(1000);  // 每秒 1000 次

Product product = redis.get("product:1001");
if (product == null) {
    if (rateLimiter.tryAcquire()) {
        product = database.query(...);
        redis.setex("product:1001", 3600, product);
    } else {
        // 返回降级数据或错误
        return getDefaultProduct();
    }
}
```

## 缓存更新策略

### 1. Cache Aside（旁路缓存）

最常用的缓存策略。

#### 读取流程

```java
public Product getProduct(int id) {
    // 1. 先查缓存
    Product product = redis.get("product:" + id);
    if (product != null) {
        return product;
    }

    // 2. 缓存未命中，查数据库
    product = database.query("SELECT * FROM products WHERE id = ?", id);

    // 3. 写入缓存
    if (product != null) {
        redis.setex("product:" + id, 3600, product);
    }

    return product;
}
```

#### 更新流程

```java
public void updateProduct(Product product) {
    // 1. 更新数据库
    database.update(product);

    // 2. 删除缓存（而不是更新缓存）
    redis.del("product:" + product.getId());
}
```

#### 为什么删除而不是更新？

- 避免并发问题
- 延迟双删解决最终一致性

### 2. Read/Write Through（穿透读写）

由缓存层负责数据库的读写。

```java
// 伪代码
class CacheLayer {
    public Product get(int id) {
        Product product = cache.get(id);
        if (product == null) {
            product = database.query(id);
            cache.set(id, product);
        }
        return product;
    }

    public void update(Product product) {
        database.update(product);
        cache.set(product.getId(), product);
    }
}
```

### 3. Write Behind（异步写入）

先更新缓存，异步批量写入数据库。

```java
public void updateProduct(Product product) {
    // 1. 更新缓存
    redis.set("product:" + product.getId(), product);

    // 2. 异步写入数据库
    queue.add(product);  // 消息队列
}

// 后台线程批量写入
while (true) {
    List<Product> products = queue.poll(100);
    database.batchUpdate(products);
}
```

## 缓存过期策略

### 1. 定时删除

设置过期时间，到期后定时删除。

```bash
SETEX key 3600 value  # 1 小时后过期
```

### 2. 惰性删除

访问键时检查是否过期，过期则删除。

```bash
GET key  # 如果键过期，返回 nil 并删除
```

### 3. 定期删除

Redis 定期随机检查设置了过期时间的键，删除过期键。

## 内存淘汰策略

当 Redis 内存不足时，根据策略淘汰键。

### 配置淘汰策略

```conf
# 最大内存
maxmemory 2gb

# 淘汰策略
maxmemory-policy allkeys-lru
```

### 淘汰策略类型

| 策略                | 说明                                      |
| ------------------- | ----------------------------------------- |
| **noeviction**      | 不淘汰，内存不足时写入报错（默认）        |
| **allkeys-lru**     | 在所有键中使用 LRU 算法淘汰               |
| **allkeys-lfu**     | 在所有键中使用 LFU 算法淘汰（Redis 4.0+） |
| **allkeys-random**  | 在所有键中随机淘汰                        |
| **volatile-lru**    | 在设置了过期时间的键中使用 LRU 淘汰       |
| **volatile-lfu**    | 在设置了过期时间的键中使用 LFU 淘汰       |
| **volatile-random** | 在设置了过期时间的键中随机淘汰            |
| **volatile-ttl**    | 淘汰即将过期的键                          |

### 推荐配置

```conf
# 通用场景：LRU
maxmemory-policy allkeys-lru

# 缓存场景：优先淘汰即将过期的键
maxmemory-policy volatile-ttl

# Redis 4.0+：LFU（访问频率）
maxmemory-policy allkeys-lfu
```

## 分布式锁

### 1. 基本实现

```bash
# 获取锁（NX: 不存在时设置，EX: 过期时间）
SET lock:resource unique_value NX EX 30

# 释放锁（使用 Lua 脚本保证原子性）
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

### 2. Java 实现

```java
public class RedisLock {
    private Jedis jedis;

    public boolean lock(String key, String value, int expireSeconds) {
        String result = jedis.set(key, value, "NX", "EX", expireSeconds);
        return "OK".equals(result);
    }

    public boolean unlock(String key, String value) {
        String script =
            "if redis.call('get', KEYS[1]) == ARGV[1] then " +
            "    return redis.call('del', KEYS[1]) " +
            "else " +
            "    return 0 " +
            "end";

        Object result = jedis.eval(script, 1, key, value);
        return Long.valueOf(1).equals(result);
    }
}

// 使用
String lockKey = "lock:resource";
String lockValue = UUID.randomUUID().toString();

if (redisLock.lock(lockKey, lockValue, 30)) {
    try {
        // 执行业务逻辑
    } finally {
        redisLock.unlock(lockKey, lockValue);
    }
}
```

### 3. Redisson 分布式锁

```java
// 引入依赖
// org.redisson:redisson:3.20.0

RedissonClient redisson = Redisson.create(config);
RLock lock = redisson.getLock("lock:resource");

try {
    // 尝试加锁，最多等待 10 秒，锁 30 秒后自动释放
    if (lock.tryLock(10, 30, TimeUnit.SECONDS)) {
        try {
            // 执行业务逻辑
        } finally {
            lock.unlock();
        }
    }
} catch (InterruptedException e) {
    Thread.currentThread().interrupt();
}
```

### 4. 分布式锁的问题

#### 锁过期问题

锁在业务未完成时过期，导致其他线程获取锁。

**解决方案**：

- 设置合理的过期时间
- 使用看门狗机制（Redisson 自动续期）

#### 锁误删问题

线程 A 的锁过期后被删除，线程 B 获取锁，线程 A 完成后删除了线程 B 的锁。

**解决方案**：

- 使用唯一标识（UUID）
- 删除时校验标识

#### Redis 集群下的问题

主节点写入锁后宕机，从节点未同步锁数据，导致多个线程获取锁。

**解决方案**：

- 使用 RedLock 算法（多个独立 Redis 实例）
- 使用 ZooKeeper 等一致性协议的分布式锁

## 最佳实践

### 1. 缓存键设计

```bash
# 使用冒号分隔，具有层次结构
user:1001:profile
product:2001:detail
cache:article:3001

# 避免
u1
p2
a3
```

### 2. 设置合理的过期时间

```java
// 热点数据：较长过期时间
redis.setex("hot:product:1001", 3600, product);  // 1 小时

// 一般数据：中等过期时间
redis.setex("product:1001", 1800, product);  // 30 分钟

// 临时数据：较短过期时间
redis.setex("captcha:1001", 300, code);  // 5 分钟
```

### 3. 监控缓存命中率

```bash
# 查看统计信息
INFO stats

# 关键指标
# keyspace_hits: 命中次数
# keyspace_misses: 未命中次数
# 命中率 = keyspace_hits / (keyspace_hits + keyspace_misses)
```

### 4. 使用 Pipeline 批量操作

```java
Pipeline pipeline = jedis.pipelined();
for (int i = 0; i < 1000; i++) {
    pipeline.get("key" + i);
}
List<Object> results = pipeline.syncAndReturnAll();
```

## 小结

Redis 缓存策略总结：

**常见问题**：

- **缓存穿透** - 查询不存在的数据（布隆过滤器、缓存空值）
- **缓存击穿** - 热点数据过期（互斥锁、永不过期）
- **缓存雪崩** - 大量缓存同时过期（随机过期时间、多级缓存）

**更新策略**：

- **Cache Aside** - 最常用（先删缓存，再更新数据库）
- **Read/Write Through** - 缓存层负责读写
- **Write Behind** - 异步写入

**淘汰策略**：

- **allkeys-lru** - 通用场景
- **volatile-ttl** - 缓存场景
- **allkeys-lfu** - 访问频率优先

**分布式锁**：

- 使用 `SET NX EX` 实现
- Lua 脚本保证原子性
- Redisson 提供高级功能
