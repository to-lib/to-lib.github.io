---
sidebar_position: 10
title: 常见问题
---

# PostgreSQL 常见问题

收集了 PostgreSQL 使用过程中的常见问题和解决方案。

## 🔧 安装和配置

### Q1: 如何修改 PostgreSQL 端口？

**A:** 编辑 `postgresql.conf`：

```conf
port = 5433
```

重启服务：

```bash
sudo systemctl restart postgresql
```

### Q2: 如何允许远程访问？

**A:** 修改两个配置文件：

1. `postgresql.conf`：

```conf
listen_addresses = '*'
```

2. `pg_hba.conf`：

```conf
host    all    all    0.0.0.0/0    md5
```

重启服务。

### Q3: 忘记 postgres 用户密码怎么办？

**A:**

```bash
# 1. 修改 pg_hba.conf，临时允许无密码登录
local   all   postgres   trust

# 2. 重启服务
sudo systemctl restart postgresql

# 3. 修改密码
psql -U postgres
ALTER USER postgres WITH PASSWORD 'new_password';

# 4. 恢复 pg_hba.conf
local   all   postgres   md5

# 5. 再次重启
sudo systemctl restart postgresql
```

## 💾 数据库操作

### Q4: 如何查看数据库大小？

```sql
SELECT
    pg_database.datname,
    pg_size_pretty(pg_database_size(pg_database.datname)) AS size
FROM pg_database
ORDER BY pg_database_size(pg_database.datname) DESC;
```

### Q5: 如何复制数据库？

```sql
CREATE DATABASE newdb WITH TEMPLATE olddb;
```

### Q6: 删除数据库时提示有活动连接？

```sql
-- 查看活动连接
SELECT * FROM pg_stat_activity WHERE datname = 'mydb';

-- 终止所有连接
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'mydb' AND pid <> pg_backend_pid();

-- 删除数据库
DROP DATABASE mydb;
```

## 📊 性能问题

### Q7: 查询很慢，如何优化？

1. **使用 EXPLAIN 分析：**

```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE age > 25;
```

2. **创建索引：**

```sql
CREATE INDEX idx_users_age ON users(age);
```

3. **更新统计信息：**

```sql
ANALYZE users;
```

### Q8: 如何找出慢查询？

```sql
-- 启用 pg_stat_statements
CREATE EXTENSION pg_stat_statements;

-- 查看最慢的查询
SELECT
    query,
    calls,
    mean_exec_time,
    max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### Q9: 数据库占用空间越来越大？

这可能是由于死元组累积。运行 VACUUM：

```sql
VACUUM ANALYZE;

-- 查看表膨胀
SELECT
    tablename,
    n_dead_tup,
    n_live_tup,
    ROUND(n_dead_tup * 100.0 / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_ratio
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;
```

## 🔐 权限问题

### Q10: 用户无法访问表？

```sql
-- 授予权限
GRANT SELECT, INSERT, UPDATE, DELETE ON users TO myuser;

-- 授予所有表的权限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO myuser;
```

### Q11: 新创建的表用户无法访问？

```sql
-- 授予未来创建的表的权限
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT ALL ON TABLES TO myuser;
```

## 🔄 并发和锁

### Q12: 如何查看当前锁？

```sql
SELECT
    pid,
    usename,
    pg_blocking_pids(pid) as blocked_by,
    query
FROM pg_stat_activity
WHERE cardinality(pg_blocking_pids(pid)) > 0;
```

### Q13: 如何处理死锁？

PostgreSQL 会自动检测并终止死锁中的一个事务。

避免死锁：

1. 按相同顺序访问资源
2. 缩短事务时间
3. 使用较低的隔离级别

### Q14: UPDATE 长时间没有响应？

可能被其他事务锁定：

```sql
-- 查看等待的锁
SELECT * FROM pg_locks WHERE NOT granted;

-- 查看阻塞的进程
SELECT
    blocked_locks.pid AS blocked_pid,
    blocking_locks.pid AS blocking_pid,
    blocked_activity.query AS blocked_query
FROM pg_locks blocked_locks
JOIN pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
WHERE NOT blocked_locks.granted;
```

## 💡 数据操作

### Q15: 如何批量插入数据？

```sql
-- 方式 1：多行 INSERT
INSERT INTO users (name) VALUES
('Alice'),
('Bob'),
('Charlie');

-- 方式 2：COPY（最快）
COPY users (name, email) FROM '/path/to/data.csv' CSV HEADER;
```

### Q16: 如何实现 MySQL 的 INSERT ... ON DUPLICATE KEY UPDATE？

```sql
INSERT INTO users (id, name, email)
VALUES (1, 'John', 'john@example.com')
ON CONFLICT (id)
DO UPDATE SET name = EXCLUDED.name, email = EXCLUDED.email;
```

### Q17: 如何实现自增 ID？

```sql
-- 方式 1：SERIAL
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- 方式 2：IDENTITY（推荐，PostgreSQL 10+）
CREATE TABLE users (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100)
);
```

## 🔍 查询问题

### Q18: 如何实现分页？

```sql
-- LIMIT + OFFSET
SELECT * FROM users
ORDER BY id
LIMIT 10 OFFSET 20;  -- 第 3 页，每页 10 条

-- 使用主键优化（大偏移量）
SELECT * FROM users
WHERE id > 20
ORDER BY id
LIMIT 10;
```

### Q19: 如何实现行转列？

```sql
-- 使用 crosstab（需要 tablefunc 扩展）
CREATE EXTENSION tablefunc;

SELECT *
FROM crosstab(
    'SELECT user_id, month, revenue FROM sales ORDER BY 1, 2'
) AS ct(user_id INT, jan NUMERIC, feb NUMERIC, mar NUMERIC);
```

### Q20: 如何查询 JSON 字段？

```sql
-- 创建表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    data JSONB
);

-- 插入数据
INSERT INTO users (data) VALUES
('{"name": "John", "age": 30, "city": "Beijing"}');

-- 查询
SELECT data->>'name' as name FROM users;
SELECT * FROM users WHERE data->>'city' = 'Beijing';
SELECT * FROM users WHERE data @> '{"age": 30}';

-- 创建索引
CREATE INDEX idx_users_data ON users USING GIN (data);
```

## 🛠️ 维护问题

### Q21: 什么时候需要 VACUUM？

- 大量 UPDATE/DELETE 操作后
- 表膨胀明显时
- 性能下降时

```sql
-- 自动 VACUUM（推荐）
autovacuum = on  -- 在 postgresql.conf 中配置

-- 手动 VACUUM
VACUUM ANALYZE users;
```

### Q22: REINDEX 和 VACUUM 的区别？

- **VACUUM**：清理死元组，释放空间
- **REINDEX**：重建索引，修复索引膨胀

```sql
VACUUM users;        -- 清理死元组
REINDEX TABLE users; -- 重建索引
```

## 🔧 故障排查

### Q23: 连接数过多怎么办？

```sql
-- 查看当前连接数
SELECT COUNT(*) FROM pg_stat_activity;

-- 查看最大连接数
SHOW max_connections;

-- 修改最大连接数（postgresql.conf）
max_connections = 200

-- 使用连接池（推荐）
# 安装 PgBouncer
sudo apt-get install pgbouncer
```

### Q24: 磁盘空间不足？

```sql
-- 查看最大的表
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;

-- 清理日志
VACUUM FULL;  -- 慎用，会锁表

-- 清理 WAL 日志
SELECT pg_switch_wal();
```

### Q25: 如何查看错误日志？

```bash
# Ubuntu/Debian
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# 查看日志位置
psql -c "SHOW log_directory;"
psql -c "SHOW log_filename;"
```

## 📚 相关资源

- [基础概念](./basic-concepts)
- [性能优化](./performance-optimization)
- [面试题](./interview-questions)
