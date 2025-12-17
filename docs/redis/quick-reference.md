---
sidebar_position: 20
title: 快速参考
---

# Redis 快速参考

## 常用命令速查

### String 字符串

| 命令   | 说明             | 示例                      |
| ------ | ---------------- | ------------------------- |
| SET    | 设置值           | `SET key value`           |
| GET    | 获取值           | `GET key`                 |
| SETEX  | 设置值和过期时间 | `SETEX key 3600 value`    |
| SETNX  | 不存在时设置     | `SETNX key value`         |
| INCR   | 自增 1           | `INCR counter`            |
| DECR   | 自减 1           | `DECR counter`            |
| INCRBY | 增加指定值       | `INCRBY counter 10`       |
| APPEND | 追加字符串       | `APPEND key append_value` |
| STRLEN | 获取长度         | `STRLEN key`              |
| MSET   | 批量设置         | `MSET k1 v1 k2 v2`        |
| MGET   | 批量获取         | `MGET k1 k2 k3`           |

### List 列表

| 命令   | 说明         | 示例               |
| ------ | ------------ | ------------------ |
| LPUSH  | 左侧插入     | `LPUSH list value` |
| RPUSH  | 右侧插入     | `RPUSH list value` |
| LPOP   | 左侧弹出     | `LPOP list`        |
| RPOP   | 右侧弹出     | `RPOP list`        |
| LRANGE | 范围查询     | `LRANGE list 0 -1` |
| LLEN   | 获取长度     | `LLEN list`        |
| LINDEX | 获取指定位置 | `LINDEX list 0`    |
| LTRIM  | 裁剪列表     | `LTRIM list 0 99`  |
| BLPOP  | 阻塞弹出     | `BLPOP list 10`    |

### Set 集合

| 命令      | 说明         | 示例               |
| --------- | ------------ | ------------------ |
| SADD      | 添加成员     | `SADD set m1 m2`   |
| SMEMBERS  | 获取所有成员 | `SMEMBERS set`     |
| SISMEMBER | 是否是成员   | `SISMEMBER set m1` |
| SREM      | 删除成员     | `SREM set m1`      |
| SCARD     | 获取数量     | `SCARD set`        |
| SUNION    | 并集         | `SUNION s1 s2`     |
| SINTER    | 交集         | `SINTER s1 s2`     |
| SDIFF     | 差集         | `SDIFF s1 s2`      |
| SPOP      | 随机弹出     | `SPOP set`         |

### Hash 哈希

| 命令    | 说明         | 示例                     |
| ------- | ------------ | ------------------------ |
| HSET    | 设置字段     | `HSET hash f1 v1`        |
| HGET    | 获取字段     | `HGET hash f1`           |
| HMSET   | 批量设置     | `HMSET hash f1 v1 f2 v2` |
| HMGET   | 批量获取     | `HMGET hash f1 f2`       |
| HGETALL | 获取所有     | `HGETALL hash`           |
| HDEL    | 删除字段     | `HDEL hash f1`           |
| HEXISTS | 字段是否存在 | `HEXISTS hash f1`        |
| HLEN    | 字段数量     | `HLEN hash`              |
| HINCRBY | 增加值       | `HINCRBY hash f1 10`     |
| HKEYS   | 所有字段名   | `HKEYS hash`             |
| HVALS   | 所有值       | `HVALS hash`             |

### Sorted Set 有序集合

| 命令      | 说明         | 示例                  |
| --------- | ------------ | --------------------- |
| ZADD      | 添加成员     | `ZADD zset 100 m1`    |
| ZRANGE    | 范围查询     | `ZRANGE zset 0 -1`    |
| ZREVRANGE | 倒序查询     | `ZREVRANGE zset 0 -1` |
| ZRANK     | 获取排名     | `ZRANK zset m1`       |
| ZSCORE    | 获取分数     | `ZSCORE zset m1`      |
| ZREM      | 删除成员     | `ZREM zset m1`        |
| ZCARD     | 成员数量     | `ZCARD zset`          |
| ZINCRBY   | 增加分数     | `ZINCRBY zset 10 m1`  |
| ZCOUNT    | 分数范围计数 | `ZCOUNT zset 0 100`   |

### 键操作

| 命令    | 说明         | 示例                       |
| ------- | ------------ | -------------------------- |
| DEL     | 删除键       | `DEL key1 key2`            |
| EXISTS  | 检查存在     | `EXISTS key`               |
| EXPIRE  | 设置过期     | `EXPIRE key 3600`          |
| TTL     | 查看剩余时间 | `TTL key`                  |
| PERSIST | 移除过期     | `PERSIST key`              |
| KEYS    | 查找键       | `KEYS pattern`             |
| SCAN    | 扫描键       | `SCAN 0 MATCH p* COUNT 10` |
| RENAME  | 重命名       | `RENAME old new`           |
| TYPE    | 获取类型     | `TYPE key`                 |
| DUMP    | 序列化       | `DUMP key`                 |
| RESTORE | 反序列化     | `RESTORE key 0 data`       |

## 配置参数速查

### 内存配置

```bash
# 最大内存
maxmemory 2gb

# 淘汰策略
maxmemory-policy allkeys-lru

# 采样数量
maxmemory-samples 5
```

### 持久化配置

```bash
# RDB
save 900 1
save 300 10
save 60 10000

# AOF
appendonly yes
appendfsync everysec
```

### 复制配置

```bash
# 从节点
replicaof master_ip 6379
masterauth password

# 只读
replica-read-only yes
```

### 网络配置

```bash
# 绑定地址
bind 127.0.0.1

# 端口
port 6379

# 超时
timeout 300

# TCP backlog
tcp-backlog 511
```

## 性能优化清单

### ✅ 键设计

- 使用层级命名（user:1001:profile）
- 设置合理的过期时间
- 避免大 key（&lt;10KB）

### ✅ 数据结构

- 小数据用 Hash 而非多个 String
- 排行榜用 Sorted Set
- 消息队列用 List 或 Stream

### ✅ 命令使用

- 避免 KEYS，用 SCAN
- 批量操作用 Pipeline
- 复杂逻辑用 Lua 脚本

### ✅ 连接管理

- 使用连接池
- 合理设置超时
- 避免频繁连接

### ✅ 监控告警

- 监控内存使用
- 监控慢查询
- 监控命中率

## 故障排查清单

### 内存问题

```bash
# 查看内存使用
INFO memory

# 分析大key
redis-cli --bigkeys

# 查看键分布
INFO keyspace
```

### 性能问题

```bash
# 查看慢查询
SLOWLOG GET 100

# 查看命令统计
INFO commandstats

# 实时监控
MONITOR
```

### 连接问题

```bash
# 查看连接数
INFO clients

# 查看阻塞客户端
CLIENT LIST
```

## 数据类型选择

| 场景       | 推荐类型      |
| ---------- | ------------- |
| 计数器     | String (INCR) |
| 缓存对象   | Hash          |
| 消息队列   | List/Stream   |
| 去重       | Set           |
| 排行榜     | Sorted Set    |
| 布隆过滤器 | Bitmap        |
| UV 统计    | HyperLogLog   |

## 命中率计算

```bash
# 获取统计信息
INFO stats

# 计算公式
命中率 = keyspace_hits / (keyspace_hits + keyspace_misses)

# 目标：95%+
```

## 内存淘汰策略

| 策略           | 说明                  |
| -------------- | --------------------- |
| noeviction     | 不淘汰，写入报错      |
| allkeys-lru    | 所有键 LRU 淘汰       |
| allkeys-lfu    | 所有键 LFU 淘汰       |
| volatile-lru   | 有过期时间的 LRU 淘汰 |
| volatile-lfu   | 有过期时间的 LFU 淘汰 |
| allkeys-random | 所有键随机淘汰        |
| volatile-ttl   | 优先淘汰即将过期的    |

## 常见端口

| 端口  | 用途           |
| ----- | -------------- |
| 6379  | Redis 默认端口 |
| 16379 | 集群总线端口   |
| 26379 | Sentinel 端口  |

## 时间复杂度

| 命令        | 复杂度       |
| ----------- | ------------ |
| GET/SET     | O(1)         |
| LPUSH/RPUSH | O(1)         |
| LPOP/RPOP   | O(1)         |
| LRANGE      | O(N)         |
| SADD        | O(1)         |
| SMEMBERS    | O(N)         |
| HSET/HGET   | O(1)         |
| HGETALL     | O(N)         |
| ZADD        | O(log N)     |
| ZRANGE      | O(log N + M) |
| KEYS        | O(N) 🚫      |
| SCAN        | O(1) ✅      |

## 总结

本快速参考涵盖了：

- ✅ 常用命令速查表
- ✅ 配置参数参考
- ✅ 性能优化清单
- ✅ 故障排查步骤
- ✅ 数据类型选择指南

随时查阅，提高效率！
