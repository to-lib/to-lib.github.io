---
sidebar_position: 13
title: Redis 性能优化
---

# Redis 性能优化

全面的 Redis 性能优化指南，涵盖内存、网络、命令等多个方面。

## 内存优化

### 1. 选择合适的数据结构

不同数据结构的内存占用差异很大。

```bash
# ❌ 使用 String 存储对象（占用内存大）
SET user:1001 '{"name":"张三","age":25,"city":"北京"}'

# ✅ 使用 Hash 存储对象（占用内存小）
HMSET user:1001 name "张三" age 25 city "北京"
```

### 2. 压缩大值

对于大值，使用压缩算法：

```java
// 压缩后存储
String data = largeData;
byte[] compressed = compress(data);  // GZIP、LZ4 等
redis.set("key", compressed);

// 读取时解压
byte[] compressed = redis.get("key");
String data = decompress(compressed);
```

### 3. 控制集合大小

避免单个键包含过多元素：

```bash
# ❌ 单个 Hash 包含 100 万条数据
HSET big_hash field1 value1
HSET big_hash field2 value2
# ... 100 万次

# ✅ 分片存储
# hash_0: 0-9999
# hash_1: 10000-19999
HSET user:hash:0 1001 "data"
HSET user:hash:1 10001 "data"
```

### 4. 设置 maxmemory

限制 Redis 使用的最大内存：

```conf
# 最大内存 2GB
maxmemory 2gb

# 内存淘汰策略
maxmemory-policy allkeys-lru
```

### 5. 监控内存使用

```bash
# 查看内存使用情况
INFO memory

# 关键指标
# used_memory: 已使用内存
# used_memory_rss: 操作系统分配的内存
# mem_fragmentation_ratio: 内存碎片率
```

### 6. 清理无用数据

定期清理过期和无用的数据：

```bash
# 查找并删除空值
SCAN 0 MATCH * COUNT 100

# 设置合理的过期时间
EXPIRE key 3600
```

## 命令优化

### 1. 避免使用危险命令

某些命令会阻塞 Redis，影响性能。

```bash
# ❌ 慎用命令（生产环境禁用）
KEYS *           # 遍历所有键，阻塞
FLUSHALL         # 清空所有数据
FLUSHDB          # 清空当前数据库
SORT large_list  # 排序大列表

# ✅ 替代方案
SCAN 0 MATCH pattern COUNT 100  # 代替 KEYS
```

### 2. 使用 Pipeline

减少网络往返次数：

```java
// ❌ 逐个执行（1000 次网络往返）
for (int i = 0; i < 1000; i++) {
    jedis.set("key" + i, "value" + i);
}

// ✅ 使用 Pipeline（1 次网络往返）
Pipeline pipeline = jedis.pipelined();
for (int i = 0; i < 1000; i++) {
    pipeline.set("key" + i, "value" + i);
}
pipeline.sync();
```

### 3. 批量操作

使用批量命令代替多次单个操作：

```bash
# ❌ 多次 GET
GET key1
GET key2
GET key3

# ✅ 使用 MGET
MGET key1 key2 key3

# ❌ 多次 SET
SET key1 value1
SET key2 value2

# ✅ 使用 MSET
MSET key1 value1 key2 value2
```

### 4. 避免大键

大键操作会阻塞 Redis：

- String：不超过 10KB
- List、Set、Hash、Sorted Set：元素不超过 1 万

```bash
# 查找大键
redis-cli --bigkeys

# 分析大键占用
MEMORY USAGE key
```

### 5. 使用 Lua 脚本

减少网络往返，保证原子性：

```bash
# ❌ 多次往返
WATCH key
GET key
MULTI
SET key new_value
EXEC

# ✅ Lua 脚本（一次往返）
EVAL "redis.call('SET', KEYS[1], ARGV[1])" 1 key new_value
```

## 网络优化

### 1. 使用连接池

避免频繁创建连接：

```java
// Jedis 连接池
JedisPoolConfig config = new JedisPoolConfig();
config.setMaxTotal(100);
config.setMaxIdle(20);
config.setMinIdle(10);
config.setTestOnBorrow(true);

JedisPool pool = new JedisPool(config, "localhost", 6379);

// 使用
try (Jedis jedis = pool.getResource()) {
    jedis.set("key", "value");
}
```

### 2. 禁用 TCP_NODELAY

主从复制时禁用，减少网络延迟：

```conf
# 主从复制禁用 Nagle 算法
repl-disable-tcp-nodelay no
```

### 3. 调整超时时间

避免长时间等待：

```java
// 设置连接超时和读取超时
Jedis jedis = new Jedis("localhost", 6379, 2000, 5000);
// 连接超时 2 秒，读取超时 5 秒
```

### 4. 压缩数据

传输大数据时压缩：

```java
// 客户端压缩
byte[] data = compress(largeData);
jedis.set(key.getBytes(), data);
```

## 持久化优化

### 1. 选择合适的持久化方式

根据场景选择：

```conf
# 缓存场景：关闭或仅 RDB
appendonly no
save 900 1

# 数据安全场景：AOF + RDB
appendonly yes
appendfsync everysec
save 900 1
```

### 2. fork 优化

减少 fork 开销：

```conf
# 使用无盘复制（磁盘慢时）
repl-diskless-sync yes

# 优化内存分配器
# 使用 jemalloc（默认已使用）
```

### 3. AOF 重写优化

```conf
# 避免频繁重写
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# 重写时不同步（提高性能，略微降低安全性）
no-appendfsync-on-rewrite yes
```

## 慢查询分析

### 1. 配置慢查询

```conf
# 慢查询阈值（微秒，10000 = 10 毫秒）
slowlog-log-slower-than 10000

# 慢查询日志最大长度
slowlog-max-len 128
```

### 2. 查看慢查询

```bash
# 查看慢查询日志
SLOWLOG GET 10

# 慢查询日志长度
SLOWLOG LEN

# 清空慢查询日志
SLOWLOG RESET
```

### 3. 分析慢查询

```bash
127.0.0.1:6379> SLOWLOG GET 5
1) 1) (integer) 6  # 日志 ID
   2) (integer) 1670000000  # 时间戳
   3) (integer) 12000  # 执行时间（微秒）
   4) 1) "KEYS"  # 命令
      2) "*"
   5) "127.0.0.1:6379"  # 客户端地址
   6) ""
```

## 监控和诊断

### 1. INFO 命令

```bash
# 查看服务器信息
INFO

# 查看特定部分
INFO server      # 服务器信息
INFO clients     # 客户端信息
INFO memory      # 内存信息
INFO persistence # 持久化信息
INFO stats       # 统计信息
INFO replication # 复制信息
INFO cpu         # CPU 信息
INFO commandstats # 命令统计
INFO cluster     # 集群信息
INFO keyspace    # 键空间信息
```

### 2. 监控关键指标

```bash
# 内存碎片率
INFO memory | grep mem_fragmentation_ratio
# 正常：1.0-1.5，过高需要重启

# 缓存命中率
INFO stats | grep keyspace
# hits / (hits + misses)

# 客户端连接数
INFO clients | grep connected_clients

# OPS（每秒操作数）
INFO stats | grep instantaneous_ops_per_sec
```

### 3. 使用 MONITOR

实时查看 Redis 执行的命令（性能开销大，仅调试使用）：

```bash
MONITOR
```

### 4. redis-cli 工具

```bash
# 查看统计信息
redis-cli --stat

# 查看大键
redis-cli --bigkeys

# 查看内存使用
redis-cli --memkeys

# 扫描键
redis-cli --scan --pattern "user:*"
```

## 高可用优化

### 1. 主从复制优化

```conf
# 复制超时
repl-timeout 60

# 复制积压缓冲区（根据网络情况调整）
repl-backlog-size 10mb

# 禁用 TCP_NODELAY
repl-disable-tcp-nodelay no

# 从节点只读
replica-read-only yes
```

### 2. 哨兵优化

```conf
# 适当增加下线判定时间
sentinel down-after-milliseconds mymaster 30000

# 控制并行复制数量
sentinel parallel-syncs mymaster 1
```

### 3. 集群优化

```conf
# 节点超时
cluster-node-timeout 15000

# 允许从节点在主节点下线时提供读服务
cluster-allow-reads-when-down no
```

## 配置优化

### 1. 内存配置

```conf
# 最大内存
maxmemory 2gb

# 淘汰策略
maxmemory-policy allkeys-lru

# 采样数量（LRU/LFU 算法）
maxmemory-samples 5
```

### 2. 线程配置

```conf
# Redis 6.0+ 多线程 I/O
io-threads 4
io-threads-do-reads yes
```

### 3. 系统配置

**Linux 内核参数**：

```bash
# 关闭透明大页
echo never > /sys/kernel/mm/transparent_hugepage/enabled

# 修改 overcommit_memory
echo 1 > /proc/sys/vm/overcommit_memory

# 增加最大文件描述符
ulimit -n 65535
```

## 最佳实践总结

### 1. 内存优化

- 选择合适的数据结构
- 控制集合大小（分片存储）
- 设置 maxmemory 和淘汰策略
- 压缩大值

### 2. 命令优化

- 避免 KEYS、FLUSHALL 等危险命令
- 使用 Pipeline 和批量命令
- 避免大键
- 使用 Lua 脚本

### 3. 网络优化

- 使用连接池
- 批量操作减少往返
- 压缩大数据

### 4. 持久化优化

- 根据场景选择持久化方式
- 优化 fork 和 AOF 重写

### 5. 监控

- 定期查看 INFO、SLOWLOG
- 监控内存、命中率、OPS
- 使用工具分析性能

## 性能测试

### 使用 redis-benchmark

```bash
# 基准测试
redis-benchmark -h 127.0.0.1 -p 6379 -c 100 -n 100000

# 测试特定命令
redis-benchmark -t set,get -n 100000 -q

# 使用 Pipeline
redis-benchmark -t set -n 1000000 -P 16

# 测试大键
redis-benchmark -t set -d 10240 -n 100000
```

### 解读测试结果

```
SET: 85000.00 requests per second
GET: 90000.00 requests per second
```

- **QPS**：每秒查询数（requests per second）
- **延迟**：P50、P95、P99 延迟

## 小结

Redis 性能优化要点：

**内存**：

- 选择合适的数据结构
- 控制集合大小
- 设置淘汰策略

**命令**：

- 避免危险命令
- 使用 Pipeline 和批量操作
- 避免大键

**网络**：

- 使用连接池
- 减少往返次数

**监控**：

- INFO、SLOWLOG
- 内存碎片率、命中率
- redis-benchmark 测试

**配置**：

- 合理的持久化策略
- 多线程 I/O（Redis 6.0+）
- 系统内核参数调优
