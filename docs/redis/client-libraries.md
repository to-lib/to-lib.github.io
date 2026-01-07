---
sidebar_position: 19
title: 客户端库详解
---

# Redis 客户端库详解

Redis 支持多种编程语言的客户端库，本文详细介绍 Java 生态中主流的 Redis 客户端，帮助你做出正确的技术选型。

## Java 客户端对比

### 主流客户端概览

| 特性 | Jedis | Lettuce | Redisson |
|------|-------|---------|----------|
| **连接方式** | 同步阻塞 | 异步非阻塞 | 异步非阻塞 |
| **线程安全** | 否（需连接池） | 是 | 是 |
| **底层实现** | BIO | Netty | Netty |
| **性能** | 中 | 高 | 中高 |
| **功能** | 基础 | 基础 | 丰富（分布式） |
| **学习成本** | 低 | 中 | 中高 |
| **Spring 默认** | ❌ | ✅ (2.0+) | ❌ |
| **适用场景** | 简单应用 | 高并发应用 | 复杂分布式场景 |

## Jedis

### 简介

Jedis 是最早的 Java Redis 客户端，API 设计简单直观，与 Redis 命令一一对应。

### 添加依赖

```xml
<dependency>
    <groupId>redis.clients</groupId>
    <artifactId>jedis</artifactId>
    <version>5.0.0</version>
</dependency>
```

### 基本使用

```java
import redis.clients.jedis.Jedis;

public class JedisExample {
    public static void main(String[] args) {
        // 创建连接（不推荐直接使用，应使用连接池）
        try (Jedis jedis = new Jedis("localhost", 6379)) {
            // 认证
            jedis.auth("password");
            
            // String 操作
            jedis.set("key", "value");
            String value = jedis.get("key");
            
            // Hash 操作
            jedis.hset("user:1001", "name", "张三");
            Map<String, String> user = jedis.hgetAll("user:1001");
            
            // List 操作
            jedis.lpush("queue", "task1", "task2");
            String task = jedis.rpop("queue");
            
            // 过期时间
            jedis.setex("session", 3600, "session_data");
        }
    }
}
```

### 连接池配置

```java
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

public class JedisPoolExample {
    private static JedisPool pool;
    
    static {
        JedisPoolConfig config = new JedisPoolConfig();
        
        // 最大连接数
        config.setMaxTotal(100);
        
        // 最大空闲连接
        config.setMaxIdle(20);
        
        // 最小空闲连接
        config.setMinIdle(10);
        
        // 获取连接最大等待时间
        config.setMaxWaitMillis(3000);
        
        // 借出时检测有效性
        config.setTestOnBorrow(true);
        
        // 归还时检测有效性
        config.setTestOnReturn(true);
        
        // 空闲时检测有效性
        config.setTestWhileIdle(true);
        
        // 空闲检测间隔
        config.setTimeBetweenEvictionRunsMillis(30000);
        
        pool = new JedisPool(config, "localhost", 6379, 2000, "password");
    }
    
    public static void main(String[] args) {
        try (Jedis jedis = pool.getResource()) {
            jedis.set("key", "value");
            System.out.println(jedis.get("key"));
        }
    }
}
```

### Pipeline 批量操作

```java
import redis.clients.jedis.Pipeline;

try (Jedis jedis = pool.getResource()) {
    Pipeline pipeline = jedis.pipelined();
    
    // 批量写入
    for (int i = 0; i < 10000; i++) {
        pipeline.set("key" + i, "value" + i);
    }
    
    // 执行
    pipeline.sync();
    
    // 或获取结果
    // List<Object> results = pipeline.syncAndReturnAll();
}
```

### 哨兵模式

```java
import redis.clients.jedis.JedisSentinelPool;

Set<String> sentinels = new HashSet<>();
sentinels.add("192.168.1.101:26379");
sentinels.add("192.168.1.102:26379");
sentinels.add("192.168.1.103:26379");

JedisSentinelPool sentinelPool = new JedisSentinelPool(
    "mymaster",    // master name
    sentinels,
    poolConfig,
    2000,          // 连接超时
    "password"     // 密码
);

try (Jedis jedis = sentinelPool.getResource()) {
    jedis.set("key", "value");
}
```

### 集群模式

```java
import redis.clients.jedis.JedisCluster;
import redis.clients.jedis.HostAndPort;

Set<HostAndPort> nodes = new HashSet<>();
nodes.add(new HostAndPort("192.168.1.101", 7001));
nodes.add(new HostAndPort("192.168.1.102", 7002));
nodes.add(new HostAndPort("192.168.1.103", 7003));

JedisCluster cluster = new JedisCluster(nodes, 2000, 2000, 5, "password", poolConfig);

cluster.set("key", "value");
String value = cluster.get("key");
```

## Lettuce

### 简介

Lettuce 是基于 Netty 的高性能异步 Redis 客户端，Spring Boot 2.0+ 默认使用 Lettuce。

### 添加依赖

```xml
<dependency>
    <groupId>io.lettuce</groupId>
    <artifactId>lettuce-core</artifactId>
    <version>6.3.0.RELEASE</version>
</dependency>
```

### 基本使用

```java
import io.lettuce.core.RedisClient;
import io.lettuce.core.api.StatefulRedisConnection;
import io.lettuce.core.api.sync.RedisCommands;

public class LettuceExample {
    public static void main(String[] args) {
        // 创建客户端
        RedisClient client = RedisClient.create("redis://password@localhost:6379/0");
        
        // 获取连接
        try (StatefulRedisConnection<String, String> connection = client.connect()) {
            // 同步命令
            RedisCommands<String, String> sync = connection.sync();
            
            sync.set("key", "value");
            String value = sync.get("key");
            System.out.println(value);
        }
        
        // 关闭客户端
        client.shutdown();
    }
}
```

### 异步操作

```java
import io.lettuce.core.api.async.RedisAsyncCommands;
import io.lettuce.core.RedisFuture;

StatefulRedisConnection<String, String> connection = client.connect();
RedisAsyncCommands<String, String> async = connection.async();

// 异步设置
RedisFuture<String> future = async.set("key", "value");
future.thenAccept(result -> System.out.println("Set 结果: " + result));

// 异步获取
async.get("key").thenAccept(value -> System.out.println("Value: " + value));

// 等待所有完成
async.set("key1", "value1");
async.set("key2", "value2");
async.set("key3", "value3");

// 使用 CompletableFuture
CompletableFuture<String> cf = async.get("key").toCompletableFuture();
cf.thenAccept(System.out::println);
```

### 响应式操作

```java
import io.lettuce.core.api.reactive.RedisReactiveCommands;
import reactor.core.publisher.Mono;

RedisReactiveCommands<String, String> reactive = connection.reactive();

// 响应式设置
Mono<String> setMono = reactive.set("key", "value");
setMono.subscribe(result -> System.out.println("Set 结果: " + result));

// 响应式获取
reactive.get("key")
    .doOnNext(value -> System.out.println("Value: " + value))
    .subscribe();

// 批量操作
reactive.mget("key1", "key2", "key3")
    .collectList()
    .subscribe(list -> list.forEach(kv -> 
        System.out.println(kv.getKey() + ": " + kv.getValue())));
```

### 连接池配置

```java
import io.lettuce.core.support.ConnectionPoolSupport;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;
import org.apache.commons.pool2.impl.GenericObjectPool;

// 配置连接池
GenericObjectPoolConfig<StatefulRedisConnection<String, String>> config = 
    new GenericObjectPoolConfig<>();
config.setMaxTotal(100);
config.setMaxIdle(20);
config.setMinIdle(10);

// 创建连接池
GenericObjectPool<StatefulRedisConnection<String, String>> pool = 
    ConnectionPoolSupport.createGenericObjectPool(
        () -> client.connect(), 
        config
    );

// 使用连接
try (StatefulRedisConnection<String, String> connection = pool.borrowObject()) {
    connection.sync().set("key", "value");
}
```

### 集群模式

```java
import io.lettuce.core.cluster.RedisClusterClient;
import io.lettuce.core.cluster.api.StatefulRedisClusterConnection;

RedisClusterClient clusterClient = RedisClusterClient.create(
    "redis://password@192.168.1.101:7001,192.168.1.102:7002,192.168.1.103:7003"
);

StatefulRedisClusterConnection<String, String> connection = clusterClient.connect();
connection.sync().set("key", "value");
```

## Redisson

### 简介

Redisson 不仅是 Redis 客户端，更是一个分布式服务框架，提供丰富的分布式数据结构和服务。

### 添加依赖

```xml
<dependency>
    <groupId>org.redisson</groupId>
    <artifactId>redisson</artifactId>
    <version>3.25.0</version>
</dependency>
```

### 基本配置

```java
import org.redisson.Redisson;
import org.redisson.api.RedissonClient;
import org.redisson.config.Config;

// 单节点配置
Config config = new Config();
config.useSingleServer()
    .setAddress("redis://localhost:6379")
    .setPassword("password")
    .setDatabase(0)
    .setConnectionPoolSize(64)
    .setConnectionMinimumIdleSize(24)
    .setConnectTimeout(10000)
    .setTimeout(3000);

RedissonClient redisson = Redisson.create(config);
```

### YAML 配置文件

```yaml
# redisson.yaml
singleServerConfig:
  address: "redis://localhost:6379"
  password: "password"
  database: 0
  connectionPoolSize: 64
  connectionMinimumIdleSize: 24
  connectTimeout: 10000
  timeout: 3000
```

```java
Config config = Config.fromYAML(new File("redisson.yaml"));
RedissonClient redisson = Redisson.create(config);
```

### 分布式锁

```java
import org.redisson.api.RLock;

RLock lock = redisson.getLock("lock:resource");

try {
    // 尝试加锁，等待 10 秒，自动释放 30 秒
    boolean acquired = lock.tryLock(10, 30, TimeUnit.SECONDS);
    if (acquired) {
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
```

### 公平锁

```java
RLock fairLock = redisson.getFairLock("fair-lock");
fairLock.lock();
try {
    // 按照请求顺序获取锁
} finally {
    fairLock.unlock();
}
```

### 读写锁

```java
RReadWriteLock rwLock = redisson.getReadWriteLock("rw-lock");

// 读锁
rwLock.readLock().lock();
try {
    // 读操作
} finally {
    rwLock.readLock().unlock();
}

// 写锁
rwLock.writeLock().lock();
try {
    // 写操作
} finally {
    rwLock.writeLock().unlock();
}
```

### 分布式集合

```java
// Map
RMap<String, String> map = redisson.getMap("myMap");
map.put("key", "value");
String value = map.get("key");

// Set
RSet<String> set = redisson.getSet("mySet");
set.add("item1");
set.add("item2");

// List
RList<String> list = redisson.getList("myList");
list.add("item1");

// Queue
RQueue<String> queue = redisson.getQueue("myQueue");
queue.offer("task1");
String task = queue.poll();

// Deque
RDeque<String> deque = redisson.getDeque("myDeque");

// SortedSet
RSortedSet<String> sortedSet = redisson.getSortedSet("mySortedSet");
```

### 分布式限流

```java
import org.redisson.api.RRateLimiter;
import org.redisson.api.RateType;

RRateLimiter limiter = redisson.getRateLimiter("myLimiter");

// 每秒 10 个请求
limiter.trySetRate(RateType.OVERALL, 10, 1, RateIntervalUnit.SECONDS);

if (limiter.tryAcquire(1)) {
    // 允许请求
} else {
    // 限流
}
```

### 分布式信号量

```java
RSemaphore semaphore = redisson.getSemaphore("mySemaphore");
semaphore.trySetPermits(10);

semaphore.acquire();
try {
    // 最多 10 个并发
} finally {
    semaphore.release();
}
```

### 集群模式

```java
Config config = new Config();
config.useClusterServers()
    .addNodeAddress(
        "redis://192.168.1.101:7001",
        "redis://192.168.1.102:7002",
        "redis://192.168.1.103:7003"
    )
    .setPassword("password")
    .setScanInterval(2000);

RedissonClient redisson = Redisson.create(config);
```

### 哨兵模式

```java
Config config = new Config();
config.useSentinelServers()
    .setMasterName("mymaster")
    .addSentinelAddress(
        "redis://192.168.1.101:26379",
        "redis://192.168.1.102:26379",
        "redis://192.168.1.103:26379"
    )
    .setPassword("password");

RedissonClient redisson = Redisson.create(config);
```

## 选型建议

### 场景推荐

| 场景 | 推荐客户端 | 原因 |
|------|-----------|------|
| 简单 CRUD | Jedis | 简单直观，学习成本低 |
| Spring Boot 项目 | Lettuce | 默认集成，异步支持 |
| 高并发读写 | Lettuce | 非阻塞 I/O，性能好 |
| 分布式锁 | Redisson | 开箱即用，功能完善 |
| 分布式限流 | Redisson | 内置限流器 |
| 响应式编程 | Lettuce | 原生支持 Reactor |
| 微服务 | Redisson | 分布式服务支持 |

### 性能比较

在高并发场景下的性能排名（仅供参考）：

1. **Lettuce**：异步非阻塞，高吞吐
2. **Redisson**：功能开销换取便利
3. **Jedis**：同步阻塞，依赖连接池

### 迁移建议

**从 Jedis 迁移到 Lettuce**：

```java
// Jedis
Jedis jedis = pool.getResource();
jedis.set("key", "value");
jedis.close();

// Lettuce
RedisCommands<String, String> commands = connection.sync();
commands.set("key", "value");
// 连接自动管理，无需手动关闭
```

## 其他语言客户端

### Python - redis-py

```python
import redis

# 连接
r = redis.Redis(host='localhost', port=6379, password='password', decode_responses=True)

# 操作
r.set('key', 'value')
value = r.get('key')

# 连接池
pool = redis.ConnectionPool(host='localhost', port=6379, password='password')
r = redis.Redis(connection_pool=pool)
```

### Go - go-redis

```go
import "github.com/redis/go-redis/v9"

client := redis.NewClient(&redis.Options{
    Addr:     "localhost:6379",
    Password: "password",
    DB:       0,
})

ctx := context.Background()
client.Set(ctx, "key", "value", 0)
val, _ := client.Get(ctx, "key").Result()
```

### Node.js - ioredis

```javascript
const Redis = require('ioredis');

const redis = new Redis({
    host: 'localhost',
    port: 6379,
    password: 'password'
});

await redis.set('key', 'value');
const value = await redis.get('key');
```

## 小结

| 客户端 | 特点 | 适用场景 |
|--------|------|----------|
| **Jedis** | 简单直观，同步阻塞 | 简单应用、学习入门 |
| **Lettuce** | 高性能异步，线程安全 | Spring Boot、高并发 |
| **Redisson** | 分布式服务框架 | 分布式锁、限流、微服务 |

**选型原则**：

- ✅ Spring Boot 项目优先使用 Lettuce
- ✅ 需要分布式锁/限流使用 Redisson
- ✅ 简单场景可以使用 Jedis
- ✅ 高并发场景使用 Lettuce 异步特性
