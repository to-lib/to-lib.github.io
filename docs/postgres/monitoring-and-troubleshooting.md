---
sidebar_position: 101
title: 监控与排障
---

# PostgreSQL 监控与排障

## 监控的核心目标

- **慢查询**：定位最耗时/最频繁 SQL
- **锁与等待**：定位阻塞链路，避免雪崩
- **连接与资源**：连接耗尽、内存与 IO 压力
- **复制与高可用**：延迟、WAL 堆积、槽位阻塞
- **容量与膨胀**：表/索引膨胀、autovacuum 是否健康

## 基础观测：pg_stat_activity

```sql
SELECT
  pid,
  usename,
  datname,
  client_addr,
  state,
  wait_event_type,
  wait_event,
  now() - query_start AS query_age,
  left(query, 200) AS query
FROM pg_stat_activity
WHERE datname = current_database()
ORDER BY query_start NULLS LAST;
```

常用筛选：

```sql
-- 只看活跃查询
SELECT pid, now() - query_start AS age, left(query, 200) AS query
FROM pg_stat_activity
WHERE state = 'active'
ORDER BY age DESC;

-- 只看空闲但持有事务的连接（危险）
SELECT pid, now() - xact_start AS xact_age, state, left(query, 200) AS query
FROM pg_stat_activity
WHERE state = 'idle in transaction'
ORDER BY xact_age DESC;
```

## 慢查询定位

### 1) 使用 EXPLAIN ANALYZE

```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM users WHERE id = 123;
```

重点关注：

- 真实耗时与预估耗时是否偏离很大（统计信息可能不准）
- 是否出现 Seq Scan（是否缺索引/索引不可用）
- Buffers 命中/读盘情况

### 2) pg_stat_statements（推荐）

启用扩展后可以按调用次数/平均耗时/总耗时排序。

```sql
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

SELECT
  query,
  calls,
  total_exec_time,
  mean_exec_time,
  rows
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

## 锁等待与阻塞排查

### 1) 找出阻塞链路

```sql
SELECT
  blocked.pid AS blocked_pid,
  blocked.usename AS blocked_user,
  now() - blocked.query_start AS blocked_for,
  left(blocked.query, 120) AS blocked_query,
  blocker.pid AS blocker_pid,
  blocker.usename AS blocker_user,
  now() - blocker.query_start AS blocker_for,
  left(blocker.query, 120) AS blocker_query
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocker
  ON blocker.pid = ANY(pg_blocking_pids(blocked.pid))
ORDER BY blocked_for DESC;
```

### 2) 必要时终止会话

```sql
-- 先尝试取消查询
SELECT pg_cancel_backend(<pid>);

-- 不行再终止连接（会回滚事务）
SELECT pg_terminate_backend(<pid>);
```

## 连接耗尽排查

### 1) 当前连接分布

```sql
SELECT
  datname,
  usename,
  state,
  count(*)
FROM pg_stat_activity
GROUP BY datname, usename, state
ORDER BY count(*) DESC;
```

### 2) 最大连接数

```sql
SHOW max_connections;
```

常见处理思路：

- 应用侧上连接池（或使用 PgBouncer）
- 缩短事务时间，避免长事务占用连接
- 排查是否存在连接泄漏

## 复制与 WAL 堆积排查

### 1) 主库查看复制状态

```sql
SELECT client_addr, state, sent_lsn, write_lsn, flush_lsn, replay_lsn
FROM pg_stat_replication;
```

### 2) 复制槽导致 WAL 不能回收

```sql
SELECT
  slot_name,
  slot_type,
  active,
  pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) AS retained_bytes
FROM pg_replication_slots
ORDER BY retained_bytes DESC;
```

## 容量与膨胀（bloat）

### 1) Top 表/索引大小

```sql
SELECT
  relname,
  pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 20;
```

### 2) dead tuples 与 autovacuum

```sql
SELECT
  schemaname,
  relname,
  n_live_tup,
  n_dead_tup,
  last_vacuum,
  last_autovacuum,
  last_analyze,
  last_autoanalyze
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC
LIMIT 20;
```

处理思路：

- 优先让 autovacuum 正常工作（别一味关掉）
- 统计信息不准时跑 `ANALYZE`
- 表严重膨胀且允许维护窗口时考虑 `VACUUM FULL`（会锁表，慎用）

## 日志与常见报错

### 1) 建议关注的日志点

- 连接/断开日志（排查连接风暴）
- 慢查询日志（结合 `log_min_duration_statement`）
- 锁等待日志（结合 `deadlock_timeout`）

### 2) 常用配置示例

```conf
# postgresql.conf
logging_collector = on
log_line_prefix = '%m [%p] %u@%d '

# 慢查询（毫秒）
log_min_duration_statement = 500

# 死锁/锁等待
deadlock_timeout = '1s'
log_lock_waits = on
```

## 排障清单（Checklist）

- **慢**：先看 `pg_stat_statements` Top SQL，然后 `EXPLAIN (ANALYZE, BUFFERS)`
- **卡**：用 `pg_blocking_pids` 找阻塞链，确认是否长事务/DDL
- **满**：连接数是否接近 `max_connections`，是否存在大量 `idle in transaction`
- **涨**：WAL/磁盘是否暴涨，是否有复制槽卡住
- **漂**：统计信息是否过期，autovacuum 是否跟得上

## 相关资源

- [性能优化](/docs/postgres/performance-optimization)
- [锁机制](/docs/postgres/locks)
- [主从复制](/docs/postgres/replication)
