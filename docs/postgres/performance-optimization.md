---
sidebar_position: 7
title: 性能优化
---

# PostgreSQL 性能优化

性能优化是数据库使用中的重要环节，本文介绍常见的优化策略和技巧。

## 📊 性能分析

### 1. EXPLAIN 分析查询计划

```sql
-- 查看查询计划
EXPLAIN SELECT * FROM users WHERE age > 25;

-- 实际执行并分析
EXPLAIN ANALYZE SELECT * FROM users WHERE age > 25;

-- 详细输出
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM users WHERE age > 25;
```

### 2. pg_stat_statements 扩展

跟踪所有 SQL 语句的执行统计。

```sql
-- 启用扩展
CREATE EXTENSION pg_stat_statements;

-- 查看最慢的查询
SELECT
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- 重置统计
SELECT pg_stat_statements_reset();
```

## 🎯 索引优化

### 1. 创建合适的索引

```sql
-- 单列索引
CREATE INDEX idx_users_email ON users(email);

-- 复合索引
CREATE INDEX idx_users_name_age ON users(last_name, first_name);

-- 部分索引
CREATE INDEX idx_active_users ON users(username)
WHERE is_active = true;
```

### 2. 查找缺失的索引

```sql
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
AND n_distinct > 100
ORDER BY correlation;
```

## 🔧 查询优化

### 1. 避免 SELECT \*

```sql
-- ❌ 不好
SELECT * FROM users;

-- ✅ 好
SELECT id, username, email FROM users;
```

### 2. 使用 LIMIT

```sql
-- 限制返回行数
SELECT * FROM users ORDER BY created_at DESC LIMIT 10;
```

### 3. 避免子查询，使用 JOIN

```sql
-- ❌ 不好
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE total > 1000);

-- ✅ 好
SELECT DISTINCT u.*
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.total > 1000;
```

### 4. 使用 EXISTS 代替 IN

```sql
-- ❌ 不好
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders);

-- ✅ 好
SELECT * FROM users u
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);
```

## 💾 配置优化

### 1. 内存配置

```sql
-- 查看当前配置
SHOW shared_buffers;
SHOW work_mem;
SHOW maintenance_work_mem;

-- 修改配置（postgresql.conf）
shared_buffers = 256MB          # 共享缓冲区（建议为系统内存的 25%）
work_mem = 4MB                  # 每个查询的工作内存
maintenance_work_mem = 64MB     # 维护操作的内存
effective_cache_size = 1GB      # 操作系统缓存大小
```

### 2. 连接池配置

```sql
max_connections = 100           # 最大连接数
```

**推荐使用连接池：**

- PgBouncer
- Pgpool-II

### 3. WAL 配置

```sql
wal_buffers = 16MB
checkpoint_timeout = 10min
max_wal_size = 1GB
```

## 🔄 VACUUM 和 ANALYZE

### 1. VACUUM

清理死元组，回收空间。

```sql
-- 手动 VACUUM
VACUUM users;

-- FULL VACUUM（锁表，慎用）
VACUUM FULL users;

-- 自动 VACUUM（推荐）
autovacuum = on
```

### 2. ANALYZE

更新统计信息，帮助查询优化器。

```sql
-- 分析表
ANALYZE users;

-- 分析所有表
ANALYZE;

-- VACUUM + ANALYZE
VACUUM ANALYZE users;
```

### 3. 监控 VACUUM

```sql
-- 查看表的膨胀情况
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    n_dead_tup,
    n_live_tup,
    ROUND(n_dead_tup * 100.0 / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_ratio
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;
```

## 🗂️ 分区表

对大表进行分区，提升查询性能。

### 范围分区

```sql
-- 创建主表
CREATE TABLE orders (
    id SERIAL,
    user_id INT,
    total NUMERIC(10, 2),
    created_at DATE NOT NULL
) PARTITION BY RANGE (created_at);

-- 创建分区
CREATE TABLE orders_2024_q1 PARTITION OF orders
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE orders_2024_q2 PARTITION OF orders
FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- 查询（自动使用合适的分区）
SELECT * FROM orders WHERE created_at >= '2024-02-01';
```

### 列表分区

```sql
CREATE TABLE users (
    id SERIAL,
    username VARCHAR(50),
    country VARCHAR(2)
) PARTITION BY LIST (country);

CREATE TABLE users_cn PARTITION OF users FOR VALUES IN ('CN');
CREATE TABLE users_us PARTITION OF users FOR VALUES IN ('US');
```

## 📈 缓存策略

### 1. 查询结果缓存

在应用层使用 Redis 等缓存查询结果。

### 2. 物化视图

```sql
-- 创建物化视图
CREATE MATERIALIZED VIEW user_stats AS
SELECT
    DATE(created_at) as date,
    COUNT(*) as user_count
FROM users
GROUP BY DATE(created_at);

-- 刷新物化视图
REFRESH MATERIALIZED VIEW user_stats;

-- 并发刷新（不锁定）
REFRESH MATERIALIZED VIEW CONCURRENTLY user_stats;

-- 创建索引
CREATE UNIQUE INDEX idx_user_stats_date ON user_stats(date);
```

## 🎓 连接优化

### 1. 使用连接池

```bash
# 安装 PgBouncer
sudo apt-get install pgbouncer

# 配置 pgbouncer.ini
[databases]
mydb = host=localhost port=5432 dbname=mydb

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
```

### 2. 减少连接数

```python
# 使用连接池
from psycopg2 import pool

connection_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=20,
    host='localhost',
    database='mydb'
)
```

## 💡 最佳实践

### 1. 批量操作

```sql
-- ❌ 不好
INSERT INTO users (username) VALUES ('user1');
INSERT INTO users (username) VALUES ('user2');
INSERT INTO users (username) VALUES ('user3');

-- ✅ 好
INSERT INTO users (username) VALUES
('user1'),
('user2'),
('user3');

-- 或使用 COPY（最快）
COPY users (username) FROM '/path/to/data.csv' CSV;
```

### 2. 预编译语句

```python
# 使用参数化查询
cursor.execute(
    "SELECT * FROM users WHERE username = %s",
    (username,)
)
```

### 3. 避免 N+1 查询问题

```sql
-- ❌ N+1 查询
SELECT * FROM users;  -- 1 次查询
-- 然后对每个用户查询订单（N 次）
SELECT * FROM orders WHERE user_id = ?;

-- ✅ 使用 JOIN
SELECT u.*, o.*
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;
```

## 📊 监控指标

### 1. 缓存命中率

```sql
SELECT
    sum(heap_blks_read) as heap_read,
    sum(heap_blks_hit) as heap_hit,
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as ratio
FROM pg_statio_user_tables;
```

### 2. 索引使用率

```sql
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### 3. 表大小

```sql
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## 📚 相关资源

- [索引优化](/docs/postgres/indexes) - 深入了解索引
- [事务管理](/docs/postgres/transactions) - 事务和并发
- [备份恢复](/docs/postgres/backup-recovery) - 数据安全

下一节：[备份恢复](/docs/postgres/backup-recovery)
