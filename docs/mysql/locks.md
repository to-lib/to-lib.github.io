---
sidebar_position: 7
title: 锁机制详解
---

# MySQL 锁机制详解

> [!TIP] > **并发控制核心**: 锁是保证数据一致性和并发访问的关键机制。理解锁的工作原理有助于优化应用性能和避免死锁。

## 锁的分类

### 按锁粒度分类

| 锁类型 | 粒度   | 开销 | 并发度 | 引擎支持       |
| ------ | ------ | ---- | ------ | -------------- |
| 表锁   | 整表   | 低   | 低     | MyISAM, InnoDB |
| 页锁   | 数据页 | 中   | 中     | BDB            |
| 行锁   | 单行   | 高   | 高     | InnoDB         |

### 按锁模式分类

| 锁类型         | 别名 | 兼容性         | 用途                                  |
| -------------- | ---- | -------------- | ------------------------------------- |
| 共享锁（S 锁） | 读锁 | S 与 S 兼容    | SELECT ... LOCK IN SHARE MODE         |
| 排他锁（X 锁） | 写锁 | 与所有锁不兼容 | UPDATE, DELETE, SELECT ... FOR UPDATE |

## 表级锁

### 表锁操作

```sql
-- 加读锁（共享锁）
LOCK TABLES users READ;

-- 加写锁（排他锁）
LOCK TABLES users WRITE;

-- 同时锁多张表
LOCK TABLES users READ, orders WRITE;

-- 解锁
UNLOCK TABLES;
```

### 表锁特点

```sql
-- 读锁阻塞写，不阻塞读
-- 会话1
LOCK TABLES users READ;
SELECT * FROM users;  -- OK

-- 会话2
SELECT * FROM users;  -- OK
INSERT INTO users ...;  -- 阻塞

-- 写锁阻塞读写
-- 会话1
LOCK TABLES users WRITE;

-- 会话2
SELECT * FROM users;  -- 阻塞
INSERT INTO users ...;  -- 阻塞
```

## InnoDB 行锁

### 行锁类型

```sql
-- 共享锁（S锁）
SELECT * FROM users WHERE id = 1 LOCK IN SHARE MODE;
-- MySQL 8.0+ 新语法
SELECT * FROM users WHERE id = 1 FOR SHARE;

-- 排他锁（X锁）
SELECT * FROM users WHERE id = 1 FOR UPDATE;
```

### 行锁实现

InnoDB 行锁是通过**锁定索引项**来实现的：

```sql
-- 有索引：锁定索引项，精确锁定行
SELECT * FROM users WHERE id = 1 FOR UPDATE;

-- 无索引：锁定全表！
SELECT * FROM users WHERE name = '张三' FOR UPDATE;
-- 如果 name 没有索引，会锁全表

-- 解决方案：添加索引
CREATE INDEX idx_name ON users(name);
```

### 锁兼容性矩阵

| 请求\持有 | S 锁    | X 锁    |
| --------- | ------- | ------- |
| S 锁      | ✅ 兼容 | ❌ 冲突 |
| X 锁      | ❌ 冲突 | ❌ 冲突 |

## 意向锁

### 意向锁概念

意向锁是表级锁，用于快速判断表中是否有行锁。

```sql
-- 意向共享锁（IS）：事务准备给数据行加 S 锁
-- 意向排他锁（IX）：事务准备给数据行加 X 锁

-- 自动加锁，无需手动操作
SELECT * FROM users WHERE id = 1 LOCK IN SHARE MODE;
-- 先加表级 IS 锁，再加行级 S 锁

SELECT * FROM users WHERE id = 1 FOR UPDATE;
-- 先加表级 IX 锁，再加行级 X 锁
```

### 意向锁兼容性

| 请求\持有 | IS  | IX  | S   | X   |
| --------- | --- | --- | --- | --- |
| IS        | ✅  | ✅  | ✅  | ❌  |
| IX        | ✅  | ✅  | ❌  | ❌  |
| S         | ✅  | ❌  | ✅  | ❌  |
| X         | ❌  | ❌  | ❌  | ❌  |

## 间隙锁（Gap Lock）

### 间隙锁概念

锁定索引记录之间的"间隙"，防止幻读。

```sql
-- 假设 users 表中 id 有 1, 5, 10 三条记录

-- 事务1
START TRANSACTION;
SELECT * FROM users WHERE id BETWEEN 3 AND 7 FOR UPDATE;
-- 锁定间隙 (1, 5) 和 (5, 10)

-- 事务2
INSERT INTO users (id, name) VALUES (4, '张三');
-- 阻塞！因为 4 在间隙 (1, 5) 中
```

### 间隙锁规则

```sql
-- 间隙锁仅在 REPEATABLE READ 隔离级别生效
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- 可以禁用间隙锁（不推荐）
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
```

## Next-Key Lock

### 概念

Next-Key Lock = 行锁 + 间隙锁，是 InnoDB 的默认行锁算法。

```sql
-- 假设 users 表中 id 有 1, 5, 10 三条记录
-- Next-Key Lock 锁定的区间是左开右闭

SELECT * FROM users WHERE id = 5 FOR UPDATE;
-- 锁定区间 (1, 5] 和 (5, 10)
```

### 加锁规则

1. 等值查询，唯一索引：退化为行锁
2. 等值查询，非唯一索引：Next-Key Lock
3. 范围查询：Next-Key Lock

```sql
-- 唯一索引等值查询：只锁定一行
SELECT * FROM users WHERE id = 5 FOR UPDATE;
-- 锁定 id = 5 这一行

-- 非唯一索引等值查询
CREATE INDEX idx_age ON users(age);
SELECT * FROM users WHERE age = 25 FOR UPDATE;
-- 锁定所有 age=25 的行，以及相邻间隙

-- 范围查询
SELECT * FROM users WHERE id > 5 FOR UPDATE;
-- 锁定 id > 5 的所有行和间隙
```

## 元数据锁（MDL）

### MDL 概念

元数据锁用于保护表结构，防止 DDL 和 DML 冲突。

```sql
-- DML 操作获取 MDL 读锁
SELECT * FROM users;

-- DDL 操作获取 MDL 写锁
ALTER TABLE users ADD COLUMN age INT;
```

### MDL 阻塞问题

```sql
-- 会话1：长事务持有 MDL 读锁
START TRANSACTION;
SELECT * FROM users;
-- 未提交...

-- 会话2：DDL 需要 MDL 写锁
ALTER TABLE users ADD COLUMN age INT;
-- 阻塞！

-- 会话3：新的查询也被阻塞
SELECT * FROM users;
-- 阻塞！因为 DDL 在排队
```

### 解决方案

```sql
-- 查看 MDL 锁等待
SELECT * FROM performance_schema.metadata_locks;

-- 设置 DDL 超时
SET lock_wait_timeout = 10;  -- 10秒超时

-- 使用 pt-online-schema-change
pt-online-schema-change --alter "ADD COLUMN age INT" D=db,t=users
```

## 死锁

### 死锁示例

```sql
-- 事务1
START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;

-- 事务2
START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 2;

-- 事务1
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
-- 等待事务2 释放 id=2 的锁

-- 事务2
UPDATE accounts SET balance = balance + 100 WHERE id = 1;
-- 等待事务1 释放 id=1 的锁
-- 死锁！
```

### 死锁检测

```sql
-- InnoDB 自动检测死锁并回滚代价较小的事务
-- 查看死锁日志
SHOW ENGINE INNODB STATUS\G

-- 关键信息在 LATEST DETECTED DEADLOCK 部分
```

### 避免死锁

```sql
-- 1. 按相同顺序访问资源
-- ✅ 正确：按 ID 升序访问
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- ❌ 错误：不同会话访问顺序不同

-- 2. 减小事务范围
-- ❌ 大事务
START TRANSACTION;
-- ... 大量操作
COMMIT;

-- ✅ 小事务
START TRANSACTION;
-- 只处理必要操作
COMMIT;

-- 3. 设置锁等待超时
SET innodb_lock_wait_timeout = 10;  -- 10秒
```

## 锁监控

### 查看当前锁

```sql
-- 查看 InnoDB 锁信息（MySQL 8.0+）
SELECT * FROM performance_schema.data_locks;
SELECT * FROM performance_schema.data_lock_waits;

-- 查看锁等待
SELECT
    r.trx_id waiting_trx_id,
    r.trx_mysql_thread_id waiting_thread,
    r.trx_query waiting_query,
    b.trx_id blocking_trx_id,
    b.trx_mysql_thread_id blocking_thread,
    b.trx_query blocking_query
FROM information_schema.innodb_lock_waits w
INNER JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id
INNER JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_trx_id;
```

### 查看 InnoDB 状态

```sql
-- 查看 InnoDB 状态
SHOW ENGINE INNODB STATUS\G

-- 关注 TRANSACTIONS 部分
-- 可以看到锁信息和死锁信息
```

## 锁优化建议

> [!IMPORTANT] > **锁优化最佳实践**:
>
> 1. ✅ 合理使用索引，避免行锁升级为表锁
> 2. ✅ 尽量使用行锁，减少锁范围
> 3. ✅ 控制事务大小，减少锁持有时间
> 4. ✅ 按固定顺序访问表和行
> 5. ✅ 使用较低的隔离级别（如果业务允许）
> 6. ✅ 设置合理的锁等待超时
> 7. ❌ 避免长事务
> 8. ❌ 避免在事务中进行外部调用

## 实战案例

### 乐观锁实现

```sql
-- 通过版本号实现乐观锁
CREATE TABLE products (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100),
    stock INT,
    version INT DEFAULT 0
);

-- 更新时检查版本号
UPDATE products
SET stock = stock - 1, version = version + 1
WHERE id = 1 AND version = 5;

-- 影响行数为 0 表示版本冲突，需要重试
```

### 悲观锁实现

```sql
-- 使用 FOR UPDATE 加悲观锁
START TRANSACTION;

SELECT * FROM products WHERE id = 1 FOR UPDATE;

-- 检查库存
-- 扣减库存
UPDATE products SET stock = stock - 1 WHERE id = 1;

COMMIT;
```

## 总结

本文介绍了 MySQL 锁机制：

- ✅ 锁的分类：表锁、行锁、页锁
- ✅ 共享锁和排他锁
- ✅ 意向锁、间隙锁、Next-Key Lock
- ✅ 元数据锁（MDL）
- ✅ 死锁检测和避免
- ✅ 锁监控和优化

继续学习 [事务处理](/docs/mysql/transactions) 和 [性能优化](/docs/mysql/performance-optimization)！
