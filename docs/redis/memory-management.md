---
sidebar_position: 14
title: 内存管理
---

# Redis 内存管理

Redis 是内存数据库，理解内存管理对于优化性能和成本至关重要。

## 内存使用分析

### INFO memory 命令

```bash
INFO memory

# 关键指标
used_memory:1073741824          # 已使用内存（字节）
used_memory_human:1.00G         # 已使用内存（可读）
used_memory_rss:1181116416      # 操作系统分配的内存
used_memory_peak:1073741824     # 内存使用峰值
used_memory_peak_human:1.00G    # 内存使用峰值（可读）
used_memory_lua:37888           # Lua 引擎内存
mem_fragmentation_ratio:1.10    # 内存碎片率
allocator:jemalloc-5.2.1        # 内存分配器
```

### 关键指标解读

| 指标                      | 说明                | 正常范围         |
| ------------------------- | ------------------- | ---------------- |
| `used_memory`             | Redis 分配的内存    | -                |
| `used_memory_rss`         | 操作系统分配的内存  | 接近 used_memory |
| `mem_fragmentation_ratio` | 碎片率 = rss / used | 1.0 - 1.5        |
| `maxmemory`               | 配置的最大内存      | 根据实际情况     |

### 内存碎片分析

**碎片率计算**：

```
碎片率 = used_memory_rss / used_memory
```

**碎片率含义**：

- `< 1` - 使用了 swap，性能严重下降
- `1.0 - 1.5` - 正常范围
- `> 1.5` - 碎片较多，可能需要处理
- `> 2.0` - 碎片严重，需要处理

### MEMORY 命令

```bash
# 查看键的内存使用
MEMORY USAGE key

# 查看内存统计
MEMORY STATS

# 查看内存分配器信息
MEMORY MALLOC-SIZE ptr

# 清理内存碎片（Redis 4.0+）
MEMORY PURGE

# 查看内存医生诊断
MEMORY DOCTOR
```

## 内存淘汰策略

当内存达到 `maxmemory` 限制时，Redis 会根据淘汰策略删除键。

### 配置

```conf
# redis.conf

# 最大内存
maxmemory 2gb

# 淘汰策略
maxmemory-policy allkeys-lru

# LRU/LFU 采样数量（精度）
maxmemory-samples 5
```

### 策略类型

#### 1. noeviction（默认）

不淘汰，内存满后写入操作报错。

```conf
maxmemory-policy noeviction
```

**适用场景**：数据不能丢失，宁可报错也不删除。

#### 2. allkeys-lru

在所有键中使用 LRU（最近最少使用）算法淘汰。

```conf
maxmemory-policy allkeys-lru
```

**适用场景**：缓存场景，最常用。

#### 3. allkeys-lfu（Redis 4.0+）

在所有键中使用 LFU（最不经常使用）算法淘汰。

```conf
maxmemory-policy allkeys-lfu
```

**适用场景**：访问频率差异大的场景。

#### 4. volatile-lru

在设置了过期时间的键中使用 LRU 淘汰。

```conf
maxmemory-policy volatile-lru
```

**适用场景**：同时存储缓存和持久化数据。

#### 5. volatile-lfu

在设置了过期时间的键中使用 LFU 淘汰。

```conf
maxmemory-policy volatile-lfu
```

#### 6. volatile-ttl

优先淘汰即将过期的键（TTL 小的先淘汰）。

```conf
maxmemory-policy volatile-ttl
```

**适用场景**：利用过期时间控制优先级。

#### 7. volatile-random

在设置了过期时间的键中随机淘汰。

```conf
maxmemory-policy volatile-random
```

#### 8. allkeys-random

在所有键中随机淘汰。

```conf
maxmemory-policy allkeys-random
```

### 策略选择指南

| 场景              | 推荐策略     | 原因                |
| ----------------- | ------------ | ------------------- |
| 缓存（通用）      | allkeys-lru  | 淘汰最久未访问的    |
| 缓存（热点数据）  | allkeys-lfu  | 保留高频访问的      |
| 缓存 + 持久化混合 | volatile-lru | 只淘汰临时数据      |
| 数据不能丢失      | noeviction   | 报错提醒            |
| 需要精确控制      | volatile-ttl | 通过 TTL 控制优先级 |

### LRU vs LFU

| 算法 | 淘汰依据     | 适用场景     |
| ---- | ------------ | ------------ |
| LRU  | 最后访问时间 | 访问模式稳定 |
| LFU  | 访问频率     | 存在热点数据 |

**LFU 示例**：

```bash
# 热点数据即使一段时间未访问，也不会被淘汰
# 因为其历史访问频率高
SET hot_key "value"
# 被频繁访问 1000 次

SET cold_key "value"
# 只被访问 1 次

# 内存满时，cold_key 先被淘汰
```

## 大 Key 分析和处理

### 什么是大 Key

- **String** > 10 KB
- **List/Set/Hash/ZSet** > 5000 个元素

### 大 Key 的问题

1. 单次操作延迟高
2. 阻塞其他请求
3. 删除时卡顿
4. 主从复制延迟
5. 内存分布不均（集群模式）

### 发现大 Key

**1. redis-cli 工具**：

```bash
# 扫描大 Key
redis-cli --bigkeys

# 输出示例
# Biggest string found 'user:1001' has 50000 bytes
# Biggest list found 'orders' has 100000 items
# Biggest hash found 'product:info' has 50000 fields
```

**2. MEMORY USAGE 命令**：

```bash
MEMORY USAGE key
# (integer) 10240  # 字节
```

**3. DEBUG OBJECT 命令**：

```bash
DEBUG OBJECT key
# Value at:0x7f3b2c10d540 refcount:1 encoding:hashtable serializedlength:10240 lru:12345 lru_seconds_idle:10
```

**4. SCAN + MEMORY USAGE 脚本**：

```bash
#!/bin/bash
# find_big_keys.sh

cursor=0
threshold=10240  # 10KB

while true; do
    result=$(redis-cli SCAN $cursor COUNT 100)
    cursor=$(echo "$result" | head -1)
    keys=$(echo "$result" | tail -n +2)

    for key in $keys; do
        size=$(redis-cli MEMORY USAGE "$key" 2>/dev/null)
        if [ -n "$size" ] && [ "$size" -gt "$threshold" ]; then
            echo "$key: $size bytes"
        fi
    done

    if [ "$cursor" -eq 0 ]; then
        break
    fi
done
```

### 处理大 Key

**1. 拆分 Hash**：

```bash
# 原来：一个大 Hash
HSET user:all 1001 "data1"
HSET user:all 1002 "data2"
# ... 100万条

# 拆分：按 ID 分片
# user:0 存储 ID 0-9999
# user:1 存储 ID 10000-19999
HSET user:0 1001 "data1"
HSET user:1 15001 "data2"
```

**2. 拆分 List**：

```bash
# 原来：一个大 List
LPUSH logs "log1" "log2" ...

# 拆分：按时间分片
LPUSH logs:2024:01:01 "log1"
LPUSH logs:2024:01:02 "log2"
```

**3. 渐进式删除**：

```bash
# 不要直接 DEL 大 Key，会阻塞
DEL big_hash  # ❌ 可能阻塞几秒

# 使用 UNLINK（异步删除，Redis 4.0+）
UNLINK big_hash  # ✅ 后台异步删除

# 或者渐进式删除
HSCAN big_hash 0 COUNT 100  # 每次删除 100 个
HDEL big_hash field1 field2 ...
```

**4. Java 渐进式删除**：

```java
public void deleteHashGradually(Jedis jedis, String key, int batchSize) {
    String cursor = "0";
    do {
        ScanResult<Map.Entry<String, String>> result =
            jedis.hscan(key, cursor, new ScanParams().count(batchSize));

        cursor = result.getCursor();
        List<Map.Entry<String, String>> entries = result.getResult();

        if (!entries.isEmpty()) {
            String[] fields = entries.stream()
                .map(Map.Entry::getKey)
                .toArray(String[]::new);
            jedis.hdel(key, fields);
        }

        // 避免阻塞太久
        try {
            Thread.sleep(10);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    } while (!"0".equals(cursor));

    jedis.del(key);  // 删除空键
}
```

## 内存碎片整理

### 碎片产生原因

1. 频繁的写入和删除
2. 数据大小变化
3. 过期键删除

### 处理方法

**1. 重启 Redis**：

最简单的方法，但有服务中断。

```bash
# 保存数据后重启
redis-cli BGSAVE
# 等待保存完成
redis-cli SHUTDOWN
redis-server /etc/redis/redis.conf
```

**2. MEMORY PURGE 命令（Redis 4.0+）**：

```bash
MEMORY PURGE
```

**3. 活跃碎片整理（Redis 4.0+）**：

```conf
# redis.conf

# 开启活跃碎片整理
activedefrag yes

# 碎片率超过 100% 开始整理
active-defrag-ignore-bytes 100mb
active-defrag-threshold-lower 10

# 碎片率超过 200% 最大力度整理
active-defrag-threshold-upper 100

# CPU 使用限制
active-defrag-cycle-min 5
active-defrag-cycle-max 75
```

动态开启：

```bash
CONFIG SET activedefrag yes
```

## 内存优化技巧

### 1. 选择合适的数据结构

```bash
# ❌ 使用多个 String
SET user:1001:name "张三"
SET user:1001:age "25"
SET user:1001:city "北京"
# 每个键都有额外开销

# ✅ 使用 Hash
HSET user:1001 name "张三" age "25" city "北京"
# 一个键，更省内存
```

### 2. 使用短键名

```bash
# ❌ 长键名
SET user:information:personal:profile:1001 "data"

# ✅ 短键名
SET u:1001 "data"
```

### 3. 使用 intset 和 ziplist

小数据量时，Redis 会使用更节省内存的编码：

```conf
# redis.conf

# Hash 使用 ziplist 的阈值
hash-max-ziplist-entries 512
hash-max-ziplist-value 64

# List 使用 quicklist 的阈值
list-max-ziplist-size -2

# Set 使用 intset 的阈值
set-max-intset-entries 512

# ZSet 使用 ziplist 的阈值
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
```

检查编码：

```bash
OBJECT ENCODING key
# "ziplist" - 压缩列表，省内存
# "hashtable" - 哈希表，占内存
```

### 4. 压缩值

对于大值，在应用层压缩：

```java
// 压缩
String data = "large data...";
byte[] compressed = Snappy.compress(data.getBytes());
jedis.set(key.getBytes(), compressed);

// 解压
byte[] compressed = jedis.get(key.getBytes());
String data = new String(Snappy.uncompress(compressed));
```

### 5. 设置合理的过期时间

```bash
# 所有缓存都应设置过期时间
SETEX cache:user:1001 3600 "user data"

# 避免缓存永不过期导致内存增长
```

### 6. 使用 OBJECT FREQ 监控热度

```bash
# 需要 maxmemory-policy 使用 LFU
OBJECT FREQ key
# 返回访问频率
```

## 内存使用估算

### 不同类型的内存占用

| 类型   | 空键开销 | 每个元素      |
| ------ | -------- | ------------- |
| String | ~90 字节 | 值大小 + 少量 |
| List   | ~88 字节 | ~24 字节/元素 |
| Set    | ~88 字节 | ~40 字节/元素 |
| Hash   | ~88 字节 | ~48 字节/字段 |
| ZSet   | ~88 字节 | ~56 字节/元素 |

### 估算公式

```
总内存 ≈ 键数量 × (键开销 + 平均值大小)
```

### 示例

存储 100 万用户信息（每个用户约 500 字节）：

```
估算内存 = 1,000,000 × (90 + 500) ≈ 560 MB
实际内存 ≈ 560 × 1.2（碎片） ≈ 672 MB
```

## 监控和告警

### 监控指标

```bash
# 内存使用率
used_memory / maxmemory

# 内存碎片率
mem_fragmentation_ratio

# 淘汰键数量
evicted_keys

# 过期键数量
expired_keys
```

### 告警阈值

| 指标       | 警告  | 严重     |
| ---------- | ----- | -------- |
| 内存使用率 | > 70% | > 90%    |
| 碎片率     | > 1.5 | > 2.0    |
| 淘汰键增长 | -     | 持续增长 |

### Prometheus 监控

```yaml
# prometheus.yml
- job_name: "redis"
  static_configs:
    - targets: ["localhost:9121"] # redis_exporter
```

```promql
# 内存使用率
redis_memory_used_bytes / redis_memory_max_bytes

# 碎片率
redis_memory_used_rss_bytes / redis_memory_used_bytes
```

## 最佳实践总结

### 1. 配置建议

```conf
# redis.conf

# 设置最大内存
maxmemory 2gb

# 使用 LRU 淘汰策略
maxmemory-policy allkeys-lru

# 开启惰性删除
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes

# 开启活跃碎片整理
activedefrag yes
```

### 2. 开发建议

- 使用短键名
- 选择合适的数据结构
- 设置过期时间
- 避免大 Key
- 压缩大值

### 3. 运维建议

- 监控内存使用
- 定期检查大 Key
- 监控碎片率
- 配置内存告警

## 小结

Redis 内存管理要点：

| 方面     | 要点                         |
| -------- | ---------------------------- |
| 监控     | INFO memory、MEMORY 命令     |
| 淘汰策略 | allkeys-lru 适合大多数场景   |
| 大 Key   | 发现、拆分、渐进式删除       |
| 碎片     | 主动整理或重启               |
| 优化     | 短键名、合适的数据结构、压缩 |

掌握内存管理，让 Redis 高效运行！
