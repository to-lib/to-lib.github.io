---
sidebar_position: 19
title: 最佳实践
---

# Redis 最佳实践

本文总结 Redis 在生产环境中的最佳实践。

## 键设计规范

### 命名规范

```bash
# 推荐：使用层级结构，用冒号分隔
user:1001:profile
product:sku:12345:info
cache:api:user:list

# 不推荐：无结构
userprofile1001
product_12345
```

### 键长度

```bash
# 好：简洁明了
u:1001:p

# 可以：可读性好
user:1001:profile

# 避免：过长影响性能
user:information:personal:detailed:profile:1001
```

## 数据结构选择

### String vs Hash

```bash
# 用户信息存储

# 方式1：String（不推荐）
SET user:1001:name "Alice"
SET user:1001:age "25"
SET user:1001:email "alice@example.com"
# 问题：键过多，内存浪费

# 方式2：Hash（推荐）
HSET user:1001 name "Alice" age "25" email "alice@example.com"
# 优点：内存高效，操作方便
```

### List vs Sorted Set

```bash
# 消息列表：List
LPUSH messages:user:1001 "message1"

# 排行榜：Sorted Set
ZADD leaderboard 100 "player1"
ZADD leaderboard 200 "player2"
```

## 内存优化

### 1. 使用合适的数据类型

```bash
# 小数据用 Hash
HSET user:1001 name "Alice" age "25"

# 避免 String 拆分
# 不好
SET user:1001:name "Alice"
SET user:1001:age "25"
```

### 2. 设置过期时间

```bash
# 缓存数据必须设置过期
SETEX cache:user:1001 3600 "user data"

# Hash 整体过期
HSET user:1001 name "Alice"
EXPIRE user:1001 3600
```

### 3. 避免大 key

```bash
# 不好：一个 Hash 存储百万条数据
HSET huge:hash field1 value1
# ... 1000000 条...

# 好：分片存储
HSET hash:0 field1 value1
HSET hash:1 field1001 value1001
```

## 性能优化

### 1. 使用 Pipeline

```java
// 批量操作用 Pipeline
Pipeline pipeline = jedis.pipelined();
for (int i = 0; i < 10000; i++) {
    pipeline.set("key" + i, "value" + i);
}
pipeline.sync();
```

### 2. 避免慢查询

```bash
# 避免
KEYS *  # 阻塞，生产禁用

# 使用
SCAN 0 MATCH user:* COUNT 100  # 渐进式遍历
```

### 3. 合理使用连接池

```java
JedisPoolConfig config = new JedisPoolConfig();
config.setMaxTotal(100);  // 最大连接数
config.setMaxIdle(20);    // 最大空闲
config.setMinIdle(10);    // 最小空闲
config.setTestOnBorrow(true);

JedisPool pool = new JedisPool(config, "localhost", 6379);
```

## 安全配置

### 1. 设置密码

```bash
# redis.conf
requirepass your_strong_password_here

# 客户端连接
AUTH your_strong_password_here
```

### 2. 禁用危险命令

```bash
# redis.conf
rename-command FLUSHALL ""
rename-command FLUSHDB ""
rename-command CONFIG "CONFIG_ADMIN"
rename-command KEYS ""
```

### 3. 绑定 IP

```bash
# redis.conf
bind 127.0.0.1

# 或指定内网IP
bind 192.168.1.100
```

## 部署建议

### 1. 主从架构

```bash
# 主节点配置
bind 0.0.0.0
protected-mode yes
requirepass master_password

# 从节点配置
replicaof master_ip 6379
masterauth master_password
replica-read-only yes
```

### 2. 哨兵部署

```bash
# sentinel.conf
sentinel monitor mymaster 127.0.0.1 6379 2
sentinel auth-pass mymaster your_password
sentinel down-after-milliseconds mymaster 5000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 10000
```

### 3. 集群规划

- 最少 3 主 3 从
- 每个主节点 1-2 个从节点
- 合理分配槽位
- 节点间网络延迟&lt;1ms

## 监控告警

### 关键指标

```bash
# 内存使用
INFO memory

# 命中率
INFO stats
# keyspace_hits / (keyspace_hits + keyspace_misses)

# 慢查询
SLOWLOG GET 10

# 连接数
INFO clients
```

### 告警阈值

- 内存使用率 > 80%
- 命中率 < 90%
- 慢查询增长
- 连接数异常
- 主从延迟 > 1s

## 常见陷阱

### 1. 缓存穿透

```java
// 解决：缓存空对象
String value = redis.get(key);
if (value == null) {
    value = db.query(key);
    if (value == null) {
        redis.setex(key, 60, "NULL");  // 缓存空值
    } else {
        redis.setex(key, 3600, value);
    }
}
```

### 2. 缓存雪崩

```java
// 解决：过期时间加随机值
int ttl = 3600 + new Random().nextInt(300);  // 3600±300秒
redis.setex(key, ttl, value);
```

### 3. 热 key 问题

```java
// 解决1：本地缓存
LoadingCache<String, String> cache = CacheBuilder.newBuilder()
    .expireAfterWrite(10, TimeUnit.SECONDS)
    .build(new CacheLoader<String, String>() {
        public String load(String key) {
            return redis.get(key);
        }
    });

// 解决2：读写分离
// 从节点读取热key
```

## 总结

- ✅ 规范键命名，使用层级结构
- ✅ 选择合适的数据结构
- ✅ 设置合理的过期时间
- ✅ 使用 Pipeline 批量操作
- ✅ 配置安全措施
- ✅ 部署高可用架构
- ✅ 建立监控告警体系
