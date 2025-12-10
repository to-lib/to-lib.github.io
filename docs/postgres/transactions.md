---
sidebar_position: 6
title: 事务管理
---

# PostgreSQL 事务管理

事务是数据库管理系统执行过程中的一个逻辑单位，保证数据的一致性和完整性。

## 📖 ACID 特性

PostgreSQL 完全支持 ACID 特性：

### 1. 原子性（Atomicity）

事务中的所有操作要么全部成功，要么全部失败。

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
-- 如果任何一条失败，都会回滚
COMMIT;
```

### 2. 一致性（Consistency）

事务执行前后，数据保持一致性状态。

```sql
-- 通过约束保证一致性
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    balance NUMERIC(10, 2) CHECK (balance >= 0)
);

BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- 如果余额不足，违反 CHECK 约束，事务回滚
COMMIT;
```

### 3. 隔离性（Isolation）

并发事务之间相互隔离，互不干扰。

### 4. 持久性（Durability）

已提交的事务永久保存，即使系统崩溃也不会丢失。

## 🎯 事务基本操作

### 开始事务

```sql
-- 方式 1
BEGIN;

-- 方式 2
START TRANSACTION;

-- 指定事务特性
BEGIN ISOLATION LEVEL SERIALIZABLE;
BEGIN READ ONLY;
```

### 提交事务

```sql
COMMIT;
```

### 回滚事务

```sql
ROLLBACK;
```

### 保存点（Savepoint）

在事务内部设置保存点，可以部分回滚。

```sql
BEGIN;

INSERT INTO users (username) VALUES ('alice');

SAVEPOINT sp1;

INSERT INTO users (username) VALUES ('bob');

ROLLBACK TO sp1;  -- 回滚到 sp1，bob 未插入

INSERT INTO users (username) VALUES ('charlie');

COMMIT;  -- alice 和 charlie 被插入
```

## 🔒 隔离级别

PostgreSQL 支持四种隔离级别：

### 1. Read Uncommitted（读未提交）

**说明**：可以读取其他事务未提交的数据（脏读）。

**注意**：PostgreSQL 实际上将其当作 Read Committed 处理。

```sql
BEGIN ISOLATION LEVEL READ UNCOMMITTED;
SELECT * FROM users;
COMMIT;
```

### 2. Read Committed（读已提交）- 默认

**说明**：只能读取已提交的数据，避免脏读。

```sql
BEGIN ISOLATION LEVEL READ COMMITTED;
SELECT * FROM users WHERE id = 1;
-- 其他事务可以修改并提交
SELECT * FROM users WHERE id = 1;  -- 可能得到不同结果
COMMIT;
```

**特点**：

- ✅ 避免脏读
- ❌ 可能出现不可重复读
- ❌ 可能出现幻读

### 3. Repeatable Read（可重复读）

**说明**：同一事务中多次读取相同数据，结果一致。

```sql
BEGIN ISOLATION LEVEL REPEATABLE READ;
SELECT * FROM users WHERE id = 1;
-- 其他事务修改并提交
SELECT * FROM users WHERE id = 1;  -- 结果与第一次相同
COMMIT;
```

**特点**：

- ✅ 避免脏读
- ✅ 避免不可重复读
- ✅ 在 PostgreSQL 中也避免幻读（通过 MVCC）

### 4. Serializable（可串行化）

**说明**：最高隔离级别，事务串行执行。

```sql
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT SUM(balance) FROM accounts;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;  -- 如果有冲突，可能失败
```

**特点**：

- ✅ 避免所有并发问题
- ❌ 性能最差
- ❌ 可能出现序列化失败

### 隔离级别对比

| 隔离级别         | 脏读 | 不可重复读 | 幻读 | 性能 |
| ---------------- | ---- | ---------- | ---- | ---- |
| Read Uncommitted | 可能 | 可能       | 可能 | 最高 |
| Read Committed   | -    | 可能       | 可能 | 高   |
| Repeatable Read  | -    | -          | -    | 中   |
| Serializable     | -    | -          | -    | 最低 |

## 🔄 MVCC（多版本并发控制）

PostgreSQL 使用 MVCC 实现并发控制。

### 工作原理

每个事务看到的是数据的一个快照，而不是锁定数据。

```sql
-- 事务 1
BEGIN;
SELECT * FROM users WHERE id = 1;  -- 看到版本 V1
-- 等待...

-- 事务 2（同时进行）
BEGIN;
UPDATE users SET age = 30 WHERE id = 1;  -- 创建版本 V2
COMMIT;

-- 事务 1 继续
SELECT * FROM users WHERE id = 1;  -- 仍然看到 V1（Repeatable Read）
COMMIT;
```

### 优势

- **高并发**：读不阻塞写，写不阻塞读
- **无锁读取**：SELECT 不需要锁
- **快照隔离**：每个事务有一致的数据视图

### 清理旧版本

```sql
-- 手动清理
VACUUM users;

-- 自动清理（推荐）
-- PostgreSQL 会自动运行 autovacuum
```

## 🔐 锁机制

### 表级锁

```sql
-- 访问共享锁（允许其他事务读）
BEGIN;
LOCK TABLE users IN ACCESS SHARE MODE;
SELECT * FROM users;
COMMIT;

-- 排他锁（阻止其他事务读写）
BEGIN;
LOCK TABLE users IN ACCESS EXCLUSIVE MODE;
-- 执行维护操作
COMMIT;
```

### 行级锁

```sql
-- FOR UPDATE - 排他锁
BEGIN;
SELECT * FROM users WHERE id = 1 FOR UPDATE;
UPDATE users SET age = 30 WHERE id = 1;
COMMIT;

-- FOR SHARE - 共享锁
BEGIN;
SELECT * FROM users WHERE id = 1 FOR SHARE;
-- 其他事务可以读，但不能修改
COMMIT;

-- SKIP LOCKED - 跳过已锁定的行
SELECT * FROM jobs WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

### 死锁

当两个或多个事务相互等待对方释放锁时，发生死锁。

```sql
-- 事务 1
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- 等待...
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- 事务 2（同时进行）
BEGIN;
UPDATE accounts SET balance = balance - 50 WHERE id = 2;
-- 等待...
UPDATE accounts SET balance = balance + 50 WHERE id = 1;
COMMIT;

-- PostgreSQL 会检测死锁并终止其中一个事务
```

**避免死锁：**

1. 按相同顺序访问资源
2. 缩短事务时间
3. 使用较低的隔离级别

## 💡 最佳实践

### 1. 保持事务简短

```sql
-- ❌ 不好
BEGIN;
SELECT * FROM large_table;  -- 长时间查询
-- 用户输入...
UPDATE users SET name = 'new_name' WHERE id = 1;
COMMIT;

-- ✅ 好
SELECT * FROM large_table;  -- 在事务外查询
BEGIN;
UPDATE users SET name = 'new_name' WHERE id = 1;
COMMIT;
```

### 2. 选择合适的隔离级别

```sql
-- 一般业务：Read Committed（默认）
BEGIN;  -- 默认 Read Committed

-- 需要一致性读：Repeatable Read
BEGIN ISOLATION LEVEL REPEATABLE READ;

-- 严格序列化：Serializable
BEGIN ISOLATION LEVEL SERIALIZABLE;
```

### 3. 错误处理

```sql
DO $$
BEGIN
    BEGIN
        UPDATE accounts SET balance = balance - 100 WHERE id = 1;
        UPDATE accounts SET balance = balance + 100 WHERE id = 2;
    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'Transaction failed: %', SQLERRM;
            ROLLBACK;
    END;
END $$;
```

### 4. 使用 CTE 和 RETURNING

```sql
-- 原子性更新多个表
WITH moved_rows AS (
    DELETE FROM pending_orders
    WHERE status = 'completed'
    RETURNING *
)
INSERT INTO completed_orders
SELECT * FROM moved_rows;
```

## 🎓 实战示例

### 转账操作

```sql
CREATE OR REPLACE FUNCTION transfer(
    from_account INT,
    to_account INT,
    amount NUMERIC
) RETURNS VOID AS $$
BEGIN
    -- 检查余额
    IF (SELECT balance FROM accounts WHERE id = from_account) < amount THEN
        RAISE EXCEPTION 'Insufficient funds';
    END IF;

    -- 扣款
    UPDATE accounts SET balance = balance - amount WHERE id = from_account;

    -- 入账
    UPDATE accounts SET balance = balance + amount WHERE id = to_account;

    -- 记录日志
    INSERT INTO transfer_logs (from_id, to_id, amount, created_at)
    VALUES (from_account, to_account, amount, NOW());
END;
$$ LANGUAGE plpgsql;

-- 使用
BEGIN;
SELECT transfer(1, 2, 100.00);
COMMIT;
```

### 分布式锁

```sql
-- 获取锁
SELECT pg_try_advisory_lock(123);

-- 执行操作
UPDATE config SET value = 'new_value' WHERE key = 'setting';

-- 释放锁
SELECT pg_advisory_unlock(123);
```

## 📊 监控事务

```sql
-- 查看当前活动事务
SELECT
    pid,
    usename,
    state,
    query,
    backend_start,
    xact_start
FROM pg_stat_activity
WHERE state != 'idle';

-- 查看长时间运行的事务
SELECT
    pid,
    now() - xact_start AS duration,
    query
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
ORDER BY xact_start;

-- 终止事务
SELECT pg_terminate_backend(pid);
```

## 📚 相关资源

- [索引优化](./indexes) - 提升查询性能
- [性能优化](./performance-optimization) - 全面优化
- [并发控制](./concurrency) - 深入了解并发

下一节：[存储过程](./stored-procedures)
