---
sidebar_position: 6
title: 锁机制
---

# PostgreSQL 锁机制

锁是数据库并发控制的核心机制，用于保证数据的一致性和完整性。

## 📚 锁的分类

| 类型       | 描述       | 粒度   |
| ---------- | ---------- | ------ |
| **表级锁** | 锁定整个表 | 粗粒度 |
| **行级锁** | 锁定特定行 | 细粒度 |

## 🔐 表级锁

### 锁模式

| 锁模式                     | 描述                              |
| -------------------------- | --------------------------------- |
| **ACCESS SHARE**           | SELECT 语句获取                   |
| **ROW SHARE**              | SELECT FOR UPDATE/SHARE           |
| **ROW EXCLUSIVE**          | UPDATE, DELETE, INSERT            |
| **SHARE UPDATE EXCLUSIVE** | VACUUM, CREATE INDEX CONCURRENTLY |
| **SHARE**                  | CREATE INDEX（非并发）            |
| **ACCESS EXCLUSIVE**       | ALTER TABLE, DROP TABLE, TRUNCATE |

### 获取表锁

```sql
LOCK TABLE users IN ACCESS SHARE MODE;
LOCK TABLE users IN SHARE MODE;
LOCK TABLE users IN ACCESS EXCLUSIVE MODE;

-- 不等待获取锁
LOCK TABLE users IN SHARE MODE NOWAIT;
```

## 🔒 行级锁

### 锁模式

| 锁模式                | SQL 语法                       | 描述       |
| --------------------- | ------------------------------ | ---------- |
| **FOR UPDATE**        | `SELECT ... FOR UPDATE`        | 排他锁     |
| **FOR NO KEY UPDATE** | `SELECT ... FOR NO KEY UPDATE` | 较弱排他锁 |
| **FOR SHARE**         | `SELECT ... FOR SHARE`         | 共享锁     |
| **FOR KEY SHARE**     | `SELECT ... FOR KEY SHARE`     | 最弱共享锁 |

### 使用示例

```sql
-- 悲观锁
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;

-- 跳过已锁定的行
SELECT * FROM jobs
WHERE status = 'pending'
LIMIT 1
FOR UPDATE SKIP LOCKED;

-- 不等待锁
SELECT * FROM users WHERE id = 1 FOR UPDATE NOWAIT;
```

### 锁等待超时

```sql
SET lock_timeout = '5s';
```

## 🔄 死锁处理

### 死锁示例

```sql
-- 事务 1
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- 事务 2（同时）
BEGIN;
UPDATE accounts SET balance = balance - 50 WHERE id = 2;
UPDATE accounts SET balance = balance + 50 WHERE id = 1;  -- 死锁！
```

### 避免死锁

1. **按相同顺序访问资源**
2. **使用 SKIP LOCKED**
3. **缩短事务时间**

```sql
-- 按 ID 顺序更新
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

## 📊 锁监控

```sql
-- 查看当前锁
SELECT l.locktype, l.relation::regclass, l.mode, l.granted, l.pid, a.query
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE l.relation IS NOT NULL;

-- 查看被阻塞的进程
SELECT pid, pg_blocking_pids(pid) AS blocked_by, query
FROM pg_stat_activity
WHERE cardinality(pg_blocking_pids(pid)) > 0;

-- 终止阻塞进程
SELECT pg_terminate_backend(pid);
```

## 🔧 Advisory Locks（咨询锁）

```sql
-- 会话级锁
SELECT pg_advisory_lock(12345);
SELECT pg_try_advisory_lock(12345);
SELECT pg_advisory_unlock(12345);

-- 事务级锁（事务结束自动释放）
SELECT pg_advisory_xact_lock(12345);
```

## 💡 最佳实践

1. **使用行级锁而非表级锁**
2. **保持事务简短**
3. **按固定顺序访问资源**
4. **使用 SKIP LOCKED 处理任务队列**
5. **设置锁超时**
6. **监控锁等待**

## 📚 相关资源

- [事务管理](/docs/postgres/transactions)
- [性能优化](/docs/postgres/performance-optimization)
