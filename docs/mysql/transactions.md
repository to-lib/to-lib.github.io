---
sidebar_position: 6
title: 事务处理
---

# MySQL 事务处理

> [!TIP] > **数据一致性保障**: 事务是保证数据一致性和完整性的核心机制。理解 ACID 特性、隔离级别和锁机制是掌握 MySQL 的关键。

## 事务基础

### 什么是事务

事务是一组 SQL 操作的集合，要么全部成功，要么全部失败。经典例子：银行转账。

```sql
-- 转账示例
START TRANSACTION;

UPDATE accounts SET balance = balance - 100 WHERE user_id = 1;  -- 扣款
UPDATE accounts SET balance = balance + 100 WHERE user_id = 2;  -- 入账

COMMIT;  -- 提交事务
```

### ACID 特性

#### Atomicity (原子性)

事务是不可分割的最小单位，要么全部成功，要么全部失败。

```sql
START TRANSACTION;

INSERT INTO orders (user_id, amount) VALUES (1, 100);
INSERT INTO order_items (order_id, product_id) VALUES (LAST_INSERT_ID(), 10);

-- 如果任何一条失败，整个事务回滚
ROLLBACK;  -- 或 COMMIT;
```

#### Consistency (一致性)

事务前后，数据保持一致状态。

```sql
-- 转账前后，总金额不变
-- A: 1000, B: 500, Total: 1500
-- A: 900,  B: 600, Total: 1500 (一致)
```

#### Isolation (隔离性)

并发事务之间相互隔离，互不干扰。

```sql
-- 事务1
START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- ... 其他操作

-- 事务2 看不到事务1未提交的修改
START TRANSACTION;
SELECT balance FROM accounts WHERE id = 1;  -- 看到原始值
```

#### Durability (持久性)

事务一旦提交，修改永久保存。

```sql
COMMIT;  -- 提交后，即使数据库崩溃，数据也不会丢失
```

## 事务操作

### 开启事务

```sql
-- 方式1：显式开启
START TRANSACTION;
-- 或
BEGIN;

-- 方式2：设置自动提交
SET autocommit = 0;  -- 关闭自动提交
-- 之后每条 SQL 都在事务中
```

### 提交事务

```sql
COMMIT;  -- 提交所有修改
```

### 回滚事务

```sql
ROLLBACK;  -- 撤销所有修改
```

### 保存点

```sql
START TRANSACTION;

INSERT INTO users (username) VALUES ('张三');
SAVEPOINT sp1;  -- 保存点1

INSERT INTO users (username) VALUES ('李四');
SAVEPOINT sp2;  -- 保存点2

INSERT INTO users (username) VALUES ('王五');

ROLLBACK TO sp2;  -- 回滚到保存点2（撤销"王五"）
ROLLBACK TO sp1;  -- 回滚到保存点1（撤销"李四"和"王五"）

COMMIT;  -- 只提交"张三"
```

## 事务隔离级别

### 四大隔离级别

| 隔离级别         | 脏读 | 不可重复读 | 幻读 | 说明                   |
| ---------------- | ---- | ---------- | ---- | ---------------------- |
| READ UNCOMMITTED | ✓    | ✓          | ✓    | 读未提交               |
| READ COMMITTED   | ✗    | ✓          | ✓    | 读已提交               |
| REPEATABLE READ  | ✗    | ✗          | ✓    | 可重复读（MySQL 默认） |
| SERIALIZABLE     | ✗    | ✗          | ✗    | 串行化                 |

### 并发问题

#### 脏读 (Dirty Read)

读取到其他事务未提交的数据。

```sql
-- 事务A
START TRANSACTION;
UPDATE accounts SET balance = 1000 WHERE id = 1;
-- 未提交

-- 事务B（READ UNCOMMITTED）
START TRANSACTION;
SELECT balance FROM accounts WHERE id = 1;  -- 读到1000（脏读）

-- 事务A回滚
ROLLBACK;  -- balance 恢复原值，事务B读到的数据无效
```

#### 不可重复读 (Non-Repeatable Read)

同一事务中，多次读取同一数据结果不同。

```sql
-- 事务A
START TRANSACTION;
SELECT balance FROM accounts WHERE id = 1;  -- 读到500

-- 事务B
START TRANSACTION;
UPDATE accounts SET balance = 1000 WHERE id = 1;
COMMIT;

-- 事务A
SELECT balance FROM accounts WHERE id = 1;  -- 读到1000（不可重复读）
```

#### 幻读 (Phantom Read)

同一事务中，多次查询记录数不同。

```sql
-- 事务A
START TRANSACTION;
SELECT COUNT(*) FROM users WHERE age > 18;  -- 结果：100

-- 事务B
START TRANSACTION;
INSERT INTO users (age) VALUES (20);
COMMIT;

-- 事务A
SELECT COUNT(*) FROM users WHERE age > 18;  -- 结果：101（幻读）
```

### 设置隔离级别

```sql
-- 查看全局隔离级别
SELECT @@global.transaction_isolation;

-- 查看会话隔离级别
SELECT @@transaction_isolation;

-- 设置会话隔离级别
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- 设置全局隔离级别
SET GLOBAL TRANSACTION ISOLATION LEVEL REPEATABLE READ;
```

### 隔离级别示例

#### READ UNCOMMITTED

```sql
-- 会话1
SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
START TRANSACTION;
SELECT balance FROM accounts WHERE id = 1;  -- 500

-- 会话2
START TRANSACTION;
UPDATE accounts SET balance = 1000 WHERE id = 1;  -- 未提交

-- 会话1
SELECT balance FROM accounts WHERE id = 1;  -- 1000（脏读）
```

#### READ COMMITTED

```sql
-- 会话1
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
START TRANSACTION;
SELECT balance FROM accounts WHERE id = 1;  -- 500

-- 会话2
START TRANSACTION;
UPDATE accounts SET balance = 1000 WHERE id = 1;
COMMIT;

-- 会话1
SELECT balance FROM accounts WHERE id = 1;  -- 1000（不可重复读，但不是脏读）
```

#### REPEATABLE READ (MySQL 默认)

```sql
-- 会话1
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;
START TRANSACTION;
SELECT balance FROM accounts WHERE id = 1;  -- 500

-- 会话2
START TRANSACTION;
UPDATE accounts SET balance = 1000 WHERE id = 1;
COMMIT;

-- 会话1
SELECT balance FROM accounts WHERE id = 1;  -- 仍然是 500（可重复读）
```

## 锁机制

### 锁的分类

#### 按锁粒度分类

- **表锁** - 锁定整张表
- **行锁** - 锁定特定行（InnoDB 支持）
- **页锁** - 锁定数据页

#### 按锁模式分类

- **共享锁 (S 锁)** - 读锁，多个事务可以同时持有
- **排它锁 (X 锁)** - 写锁，独占访问

### 表锁

```sql
-- 加读锁
LOCK TABLES users READ;
SELECT * FROM users;  -- 允许
UPDATE users SET age = 25;  -- 报错（只读）
UNLOCK TABLES;

-- 加写锁
LOCK TABLES users WRITE;
SELECT * FROM users;  -- 允许
UPDATE users SET age = 25;  -- 允许
UNLOCK TABLES;
```

### 行锁 (InnoDB)

#### 共享锁 (S 锁)

```sql
-- 加共享锁
START TRANSACTION;
SELECT * FROM users WHERE id = 1 LOCK IN SHARE MODE;

-- 其他事务可以读，但不能写
-- 其他事务: SELECT * FROM users WHERE id = 1;  -- 成功
-- 其他事务: UPDATE users SET age = 25 WHERE id = 1;  -- 等待
```

#### 排它锁 (X 锁)

```sql
-- 加排它锁
START TRANSACTION;
SELECT * FROM users WHERE id = 1 FOR UPDATE;

-- 其他事务不能读（加锁读）也不能写
-- 其他事务: SELECT * FROM users WHERE id = 1 FOR UPDATE;  -- 等待
-- 其他事务: UPDATE users SET age = 25 WHERE id = 1;  -- 等待
```

#### 意向锁

- **意向共享锁 (IS)**
- **意向排它锁 (IX)**

```sql
-- InnoDB 自动加意向锁
-- 行锁之前先加表级意向锁
```

### 间隙锁 (Gap Lock)

防止幻读，锁定索引范围。

```sql
-- REPEATABLE READ 隔离级别下
START TRANSACTION;
SELECT * FROM users WHERE id BETWEEN 10 AND 20 FOR UPDATE;

-- 锁定 (10, 20) 范围，其他事务无法插入 id=15 的记录
-- 其他事务: INSERT INTO users (id, username) VALUES (15, '张三');  -- 等待
```

### Next-Key Lock

行锁 + 间隙锁，InnoDB 默认使用。

```sql
-- 锁定记录 + 记录前的间隙
START TRANSACTION;
SELECT * FROM users WHERE id >= 10 FOR UPDATE;
-- 锁定 id=10 的记录 + (前一条记录, 10] 的间隙
```

## 死锁

### 什么是死锁

两个或多个事务互相等待对方释放锁，形成循环等待。

```sql
-- 事务A
START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- 锁定 id=1
-- 等待锁定 id=2
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- 事务B
START TRANSACTION;
UPDATE accounts SET balance = balance - 50 WHERE id = 2;  -- 锁定 id=2
-- 等待锁定 id=1
UPDATE accounts SET balance = balance + 50 WHERE id = 1;

-- 死锁！MySQL 会自动检测并回滚其中一个事务
```

### 查看死锁

```sql
-- 查看最近一次 InnoDB 死锁信息
SHOW ENGINE INNODB STATUS\G

-- 查看锁等待情况
SELECT * FROM performance_schema.data_locks;
SELECT * FROM performance_schema.data_lock_waits;
```

### 避免死锁

> [!IMPORTANT] > **死锁预防策略**:
>
> 1. 按相同顺序访问表和行
> 2. 尽量使用索引访问数据
> 3. 减少事务持有锁的时间
> 4. 使用较低的隔离级别
> 5. 合理设置锁等待超时

```sql
-- 设置锁等待超时
SET innodb_lock_wait_timeout = 50;  -- 50秒

-- 统一访问顺序
-- ✅ 好：按 ID 升序访问
START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- ❌ 不好：不同事务按不同顺序访问
-- 事务A: 先1后2
-- 事务B: 先2后1  -- 可能死锁
```

## MVCC (多版本并发控制)

### MVCC 原理

InnoDB 使用 MVCC 实现非锁定读，提高并发性能。

- 每行记录有多个版本
- 事务读取时根据版本号判断可见性
- 不需要加锁，提高并发

### 隐藏列

InnoDB 为每行添加隐藏列：

- **DB_TRX_ID** - 事务 ID
- **DB_ROLL_PTR** - 回滚指针
- **DB_ROW_ID** - 行 ID（无主键时）

### Read View

事务开始时创建 Read View，确定哪些版本可见。

```sql
-- REPEATABLE READ：事务开始时创建 Read View
START TRANSACTION;
SELECT * FROM users WHERE id = 1;  -- 创建 Read View
-- 之后的查询使用同一个 Read View（可重复读）

-- READ COMMITTED：每次查询都创建新的 Read View
START TRANSACTION;
SELECT * FROM users WHERE id = 1;  -- 创建 Read View 1
SELECT * FROM users WHERE id = 1;  -- 创建 Read View 2
```

## 事务日志

### Undo Log (回滚日志)

用于回滚和 MVCC。

```sql
-- 事务开始
START TRANSACTION;
UPDATE users SET age = 26 WHERE id = 1;  -- Undo Log: age = 25

-- 回滚
ROLLBACK;  -- 使用 Undo Log 恢复 age = 25
```

### Redo Log (重做日志)

用于崩溃恢复，保证持久性。

```sql
-- 事务提交
START TRANSACTION;
UPDATE users SET age = 26 WHERE id = 1;
COMMIT;  -- 写入 Redo Log，即使崩溃也能恢复
```

### Binlog (二进制日志)

用于主从复制和数据恢复。

```sql
-- 开启 binlog
SET GLOBAL log_bin = ON;

-- 查看 binlog 文件
SHOW BINARY LOGS;

-- 查看 binlog 内容
SHOW BINLOG EVENTS IN 'mysql-bin.000001';
```

## 最佳实践

### 事务使用规范

> [!IMPORTANT] > **事务最佳实践**:
>
> 1. 保持事务简短，减少锁持有时间
> 2. 避免在事务中执行耗时操作（如网络请求）
> 3. 合理选择隔离级别（一般用默认的 REPEATABLE READ）
> 4. 使用索引访问数据，减少锁范围
> 5. 统一访问顺序，避免死锁
> 6. 及时提交或回滚事务

### 示例：规范的事务处理

```sql
-- ✅ 好的事务
START TRANSACTION;
-- 1. 快速执行
UPDATE accounts SET balance = balance - 100 WHERE user_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE user_id = 2;
-- 2. 及时提交
COMMIT;

-- ❌ 不好的事务
START TRANSACTION;
-- 1. 耗时操作
SELECT * FROM large_table;  -- 全表扫描
-- 2. 外部调用
-- 调用外部 API（网络延迟）
-- 3. 长时间持有锁
-- ... 其他操作
COMMIT;
```

### 错误处理

```sql
-- 使用存储过程处理事务和错误
DELIMITER //

CREATE PROCEDURE transfer_money(
    IN from_user_id INT,
    IN to_user_id INT,
    IN amount DECIMAL(10,2)
)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;  -- 发生错误时回滚
    END;

    START TRANSACTION;

    UPDATE accounts SET balance = balance - amount WHERE user_id = from_user_id;
    UPDATE accounts SET balance = balance + amount WHERE user_id = to_user_id;

    COMMIT;  -- 成功时提交
END//

DELIMITER ;
```

## 总结

本文详细介绍了 MySQL 事务处理：

- ✅ ACID 特性：原子性、一致性、隔离性、持久性
- ✅ 事务操作：BEGIN、COMMIT、ROLLBACK、SAVEPOINT
- ✅ 隔离级别：READ UNCOMMITTED、READ COMMITTED、REPEATABLE READ、SERIALIZABLE
- ✅ 锁机制：表锁、行锁、间隙锁、Next-Key Lock
- ✅ 死锁检测和避免
- ✅ MVCC 多版本并发控制
- ✅ 事务日志：Undo Log、Redo Log、Binlog

掌握事务处理后，可以继续学习 [性能优化](./performance-optimization)！
