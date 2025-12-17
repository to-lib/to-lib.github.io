---
sidebar_position: 21
title: 常见问题 FAQ
---

# Redis 常见问题

## 基础问题

### Q: Redis 是什么？

**A:** Redis（Remote Dictionary Server）是一个开源的内存数据存储系统，可用作数据库、缓存、消息队列等。

**特点：**

- 内存存储，速度快
- 支持多种数据类型
- 支持持久化
- 高可用（主从、哨兵、集群）

### Q: Redis 为什么这么快？

**A:**

1. **内存操作** - 数据存储在内存中
2. **单线程模型** - 避免线程切换和锁竞争
3. **高效数据结构** - 针对性优化的数据结构
4. **I/O 多路复用** - epoll 等高效 I/O 模型

### Q: Redis 单线程为什么还这么快？

**A:**

- CPU 不是瓶颈，内存和网络才是
- 避免了多线程的上下文切换
- 避免了锁机制
- 使用 I/O 多路复用处理并发连接

### Q: Redis 支持哪些数据类型？

**A:**

**基本类型：**

- String（字符串）
- List（列表）
- Set（集合）
- Hash（哈希）
- Sorted Set（有序集合）

**高级类型：**

- Bitmap（位图）
- HyperLogLog（基数统计）
- Geo（地理位置）
- Stream（数据流）

### Q: String 和 Hash 什么时候用？

**A:**

**String 适合：**

```bash
# 简单键值对
SET counter 100
SET cache:user:1001 "user data json"
```

**Hash 适合：**

```bash
# 对象存储
HSET user:1001 name "Alice" age "25" email "alice@example.com"
```

**选择建议：**

- 单个值用 String
- 对象/多字段用 Hash

## 持久化问题

### Q: RDB 和 AOF 有什么区别？

**A:**

| 特性     | RDB          | AOF      |
| -------- | ------------ | -------- |
| 文件大小 | 小           | 大       |
| 恢复速度 | 快           | 慢       |
| 数据安全 | 可能丢失数据 | 安全性高 |
| 性能影响 | 小           | 相对大   |

**建议：** 同时开启 RDB 和 AOF

### Q: 如何选择持久化策略？

**A:**

```bash
# 对数据安全性要求高
appendonly yes
appendfsync everysec

# 对性能要求高，可容忍少量数据丢失
save 900 1
save 300 10
appendonly no
```

### Q: AOF 文件太大怎么办？

**A:**

```bash
# 手动重写
BGREWRITEAOF

# 自动重写配置
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

## 缓存问题

### Q: 什么是缓存穿透？如何解决？

**A:** 查询不存在的数据，每次都打到数据库。

**解决方案：**

**1. 缓存空对象**

```java
String value = redis.get(key);
if (value == null) {
    value = db.query(key);
    if (value == null) {
        redis.setex(key, 60, "NULL");  // 缓存空值
    }
}
```

**2. 布隆过滤器**

```java
if (!bloomFilter.mightContain(key)) {
    return null;  // 直接返回
}
```

### Q: 什么是缓存击穿？如何解决？

**A:** 热点 key 突然失效，大量请求打到数据库。

**解决方案：**

**1. 永不过期**

```bash
SET hotkey value
# 不设置过期时间
```

**2. 互斥锁**

```java
String value = redis.get(key);
if (value == null) {
    if (redis.setnx("lock:" + key, "1", 10)) {
        value = db.query(key);
        redis.setex(key, 3600, value);
        redis.del("lock:" + key);
    } else {
        // 等待重试
        Thread.sleep(100);
        return redis.get(key);
    }
}
```

### Q: 什么是缓存雪崩？如何解决？

**A:** 大量 key 同时失效，请求压垮数据库。

**解决方案：**

**1. 过期时间加随机值**

```java
int ttl = 3600 + new Random().nextInt(300);
redis.setex(key, ttl, value);
```

**2. 互斥锁**

```java
// 同缓存击穿
```

**3. 多级缓存**

```java
// Redis + 本地缓存
// Nginx + Redis + DB
```

## 性能问题

### Q: 如何提升 Redis 性能？

**A:**

**1. 使用 Pipeline**

```java
Pipeline p = jedis.pipelined();
for (int i = 0; i < 10000; i++) {
    p.set("k" + i, "v" + i);
}
p.sync();
```

**2. 避免大 key**

```bash
# 不好
HSET big_hash f1 v1 ... # 100万个字段

# 好：拆分
HSET hash:0 f1 v1
HSET hash:1 f1000000 v1000000
```

**3. 合理使用数据结构**

**4. 设置合理的过期时间**

### Q: KEYS 命令为什么禁用？

**A:** `KEYS *` 会阻塞 Redis，生产环境禁用。

**替代方案：**

```bash
# 使用 SCAN
SCAN 0 MATCH user:* COUNT 100
```

### Q: 如何分析慢查询？

**A:**

**1. 查看慢查询日志**

```bash
SLOWLOG GET 100
```

**2. 配置慢查询阈值**

```bash
# redis.conf
slowlog-log-slower-than 10000  # 10ms
slowlog-max-len 128
```

**3. 分析和优化**

- 避免使用 KEYS
- 减少大 key
- 优化 Lua 脚本

## 集群问题

### Q: 主从、哨兵、集群有什么区别？

**A:**

**主从复制：**

- 数据备份
- 读写分离
- 主节点故障需要手动切换

**哨兵模式：**

- 基于主从
- 自动故障转移
- 监控和通知
- 适合中小规模

**Redis 集群：**

- 数据分片
- 横向扩展
- 高可用
- 适合大规模

### Q: 如何保证主从数据一致性？

**A:**

**1. 同步复制（不推荐）**

```bash
wait 1 1000  # 等待1个从节点确认，超时1秒
```

**2. 异步复制+重试**

```java
// 应用层保证最终一致性
```

**3. 监控主从延迟**

```bash
INFO replication
# master_repl_offset
# slave_repl_offset
```

### Q: 集群如何扩容？

**A:**

```bash
# 1. 添加新节点
redis-cli --cluster add-node new_ip:6379 existing_ip:6379

# 2. 分配槽位
redis-cli --cluster reshard existing_ip:6379
```

## 内存问题

### Q: Redis 内存满了怎么办？

**A:**

**1. 查看内存使用**

```bash
INFO memory
```

**2. 设置淘汰策略**

```bash
maxmemory 2gb
maxmemory-policy allkeys-lru
```

**3. 分析大 key**

```bash
redis-cli --bigkeys
```

**4. 清理过期 key**

```bash
# 自动清理
lazyfree-lazy-eviction yes
```

### Q: 如何优化内存使用？

**A:**

**1. 使用 Hash 存储对象**

```bash
# 不好
SET user:1001:name "Alice"
SET user:1001:age "25"

# 好
HSET user:1001 name "Alice" age "25"
```

**2. 设置过期时间**

```bash
SETEX key 3600 value
```

**3. 启用压缩**

```bash
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
```

## 安全问题

### Q: 如何保证 Redis 安全？

**A:**

**1. 设置密码**

```bash
requirepass your_password
```

**2. 禁用危险命令**

```bash
rename-command FLUSHALL ""
rename-command CONFIG ""
```

**3. 绑定 IP**

```bash
bind 127.0.0.1 192.168.1.100
```

**4. 使用防火墙**

```bash
# 只允许应用服务器访问
```

### Q: Redis 被攻击了怎么办？

**A:**

**1. 立即修改密码**

```bash
CONFIG SET requirepass new_password
```

**2. 检查数据**

```bash
KEYS *
```

**3. 恢复数据**

```bash
# 从备份恢复
```

**4. 加强安全措施**

## 开发问题

### Q: Jedis vs Lettuce？

**A:**

| 特性        | Jedis      | Lettuce  |
| ----------- | ---------- | -------- |
| 线程安全    | ❌         | ✅       |
| 异步支持    | ❌         | ✅       |
| 连接池      | 需要       | 可选     |
| Spring Boot | 2.x 不推荐 | 2.x 默认 |

**建议：** 新项目用 Lettuce

### Q: 如何测试 Redis？

**A:**

**单元测试：**

```java
@Test
public void testRedis() {
    Jedis jedis = new Jedis("localhost", 6379);
    jedis.set("test", "value");
    assertEquals("value", jedis.get("test"));
}
```

**集成测试：**

```java
@SpringBootTest
@AutoConfigureDataRedis
public class RedisIntegrationTest {
    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    @Test
    public void test() {
        redisTemplate.opsForValue().set("key", "value");
    }
}
```

## 总结

- ✅ Redis 是高性能的内存数据库
- ✅ 理解不同数据类型的适用场景
- ✅ 合理配置持久化策略
- ✅ 掌握缓存常见问题的解决方案
- ✅ 关注性能和安全
- ✅ 选择合适的高可用架构

遇到问题先查文档，合理使用 Redis！
