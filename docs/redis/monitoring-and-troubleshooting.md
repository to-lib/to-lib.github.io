---
sidebar_position: 25
title: 监控与排障
---

# Redis 监控与排障

本章目标：把 Redis 的“现象”快速定位到“原因”，并给出可以落地的排查顺序。

## 关键监控指标（必备）

### 1. 可用性

```bash
redis-cli PING
redis-cli INFO server
```

关注：

- `uptime_in_seconds`
- `role`（主/从）

### 2. 吞吐与延迟

```bash
redis-cli INFO stats
redis-cli --stat
```

关注：

- `instantaneous_ops_per_sec`
- `total_commands_processed`

### 3. 命中率（缓存场景）

```bash
redis-cli INFO stats | grep -E 'keyspace_(hits|misses)'
```

计算：

- 命中率 = `keyspace_hits / (keyspace_hits + keyspace_misses)`

### 4. 内存与碎片

```bash
redis-cli INFO memory
```

关注：

- `used_memory`
- `used_memory_rss`
- `mem_fragmentation_ratio`

更多见：[内存管理](/docs/redis/memory-management)

### 5. 慢查询

```bash
redis-cli SLOWLOG LEN
redis-cli SLOWLOG GET 10
```

配置建议见：[性能优化](/docs/redis/performance-optimization)

## 常见故障场景与排查路径

## 1) 连接不上 Redis

排查顺序：

1. Redis 是否存活：`PING`
2. 端口监听：检查 `bind`/`port`
3. 认证失败：检查 `requirepass`/ACL
4. 网络/防火墙：确认应用服务器到 Redis 的连通性

相关配置见：[配置与部署](/docs/redis/configuration)

## 2) CPU 飙高 / QPS 很高

常见原因：

- 热 key 导致集中访问
- 慢命令（如 `KEYS`、大范围 `LRANGE`、大 Hash `HGETALL`）
- Lua 脚本执行过重

建议动作：

- 先看 `INFO commandstats`
- 结合 `SLOWLOG GET`
- 临时调试可以用 `MONITOR`（仅短时间）

```bash
redis-cli INFO commandstats
redis-cli SLOWLOG GET 20
```

## 3) 内存持续增长 / OOM / 频繁淘汰

排查顺序：

1. 是否设置 `maxmemory`
2. 淘汰策略是否匹配业务
3. 是否存在大 Key / 热 Key

```bash
redis-cli CONFIG GET maxmemory
redis-cli INFO stats | grep evicted_keys
redis-cli --bigkeys
```

## 4) 主从延迟 / 数据不一致

排查顺序：

1. 查看复制状态：`INFO replication`
2. 关注 offset 差异、网络抖动
3. 大 Key、全量同步、磁盘 IO 都可能导致延迟

```bash
redis-cli INFO replication
```

详见：[主从复制](/docs/redis/replication)

## 5) 集群问题（MOVED/ASK/槽位不均）

排查要点：

- 客户端是否支持 Cluster
- 节点是否有 fail 状态
- 槽位迁移是否在进行

```bash
redis-cli -c CLUSTER INFO
redis-cli -c CLUSTER NODES
```

详见：[Redis 集群](/docs/redis/cluster)

## 日志与基础诊断命令

```bash
# 全量信息（输出很长，调试时使用）
redis-cli INFO

# 客户端连接与阻塞
redis-cli INFO clients
redis-cli CLIENT LIST

# 键空间
redis-cli INFO keyspace
```

## Prometheus 监控（可选）

如果你有 Prometheus，可接入 `redis_exporter`，至少监控：

- 内存使用率
- ops
- 命中率
- 主从状态
- 慢查询增长

## 排障最小清单

- [ ] `INFO`（server/clients/memory/stats/persistence/replication/cluster）
- [ ] `SLOWLOG GET 10`
- [ ] `INFO commandstats`
- [ ] `redis-cli --bigkeys`
- [ ] 查看日志文件（loglevel/logfile）

涉及安全问题时，优先参考：[安全配置](/docs/redis/security)
