---
sidebar_position: 11
title: PostgreSQL 面试题
---

# PostgreSQL 面试题

精选 PostgreSQL 常见面试题及答案。

## 📚 基础题

### 1. PostgreSQL 与 MySQL 的主要区别？

**答案：**

| 特性          | PostgreSQL                       | MySQL                      |
| ------------- | -------------------------------- | -------------------------- |
| **ACID 支持** | 完全支持                         | 部分支持（取决于存储引擎） |
| **并发控制**  | MVCC                             | 锁机制                     |
| **复杂查询**  | 优秀（窗口函数、CTE、递归查询）  | 良好                       |
| **数据类型**  | 非常丰富（JSON、数组、范围类型） | 基本                       |
| **全文搜索**  | 内置强大的全文搜索               | 基础支持                   |
| **扩展性**    | 强大（可自定义类型、函数）       | 一般                       |

### 2. 什么是 MVCC？

**答案：**

MVCC（Multi-Version Concurrency Control，多版本并发控制）是 PostgreSQL 的并发控制机制。

**工作原理：**

- 每个事务看到的数据是一个快照
- 读操作不阻塞写操作
- 写操作不阻塞读操作
- 通过版本号判断数据可见性

**优点：**

- 高并发性能
- 读操作无需锁
- 避免了大部分锁冲突

### 3. 解释 PostgreSQL 的 ACID 特性

**答案：**

- **原子性（Atomicity）**：事务中的所有操作要么全部成功，要么全部失败
- **一致性（Consistency）**：事务执行前后，数据保持一致性状态
- **隔离性（Isolation）**：并发事务之间相互隔离，互不干扰
- **持久性（Durability）**：已提交的事务永久保存，即使系统崩溃也不丢失

## 🎯 中级题

### 4. PostgreSQL 有哪些索引类型？各适用于什么场景？

**答案：**

| 索引类型   | 适用场景                       | 示例                                               |
| ---------- | ------------------------------ | -------------------------------------------------- |
| **B-Tree** | 大多数场景，支持排序和范围查询 | `CREATE INDEX idx ON users(name)`                  |
| **Hash**   | 等值查询                       | `CREATE INDEX idx ON users USING HASH(email)`      |
| **GIN**    | 数组、JSON、全文搜索           | `CREATE INDEX idx ON users USING GIN(tags)`        |
| **GiST**   | 几何数据、全文搜索             | `CREATE INDEX idx ON locations USING GIST(coords)` |
| **BRIN**   | 大表中有自然顺序的列           | `CREATE INDEX idx ON logs USING BRIN(created_at)`  |

### 5. 什么是索引失效？如何避免？

**答案：**

**常见导致索引失效的情况：**

1. **在索引列上使用函数**

```sql
-- ❌ 索引失效
SELECT * FROM users WHERE UPPER(username) = 'JOHN';

-- ✅ 使用表达式索引
CREATE INDEX idx ON users(UPPER(username));
```

2. **使用 OR 连接不同列**

```sql
-- ❌ 可能失效
SELECT * FROM users WHERE name = 'John' OR email = 'john@example.com';

-- ✅ 使用 UNION
SELECT * FROM users WHERE name = 'John'
UNION
SELECT * FROM users WHERE email = 'john@example.com';
```

3. **违反最左前缀原则**

```sql
-- 索引：(last_name, first_name)
-- ❌ 不能使用索引
SELECT * FROM users WHERE first_name = 'John';

-- ✅ 可以使用索引
SELECT * FROM users WHERE last_name = 'Smith';
```

### 6. PostgreSQL 的隔离级别有哪些？

**答案：**

| 隔离级别                   | 脏读 | 不可重复读 | 幻读 |
| -------------------------- | ---- | ---------- | ---- |
| **Read Uncommitted**       | 可能 | 可能       | 可能 |
| **Read Committed**（默认） | -    | 可能       | 可能 |
| **Repeatable Read**        | -    | -          | -    |
| **Serializable**           | -    | -          | -    |

**注意：** PostgreSQL 的 Read Uncommitted 实际上等同于 Read Committed。

### 7. VACUUM 的作用是什么？

**答案：**

**VACUUM 的作用：**

1. 清理死元组（Dead Tuples）
2. 释放磁盘空间
3. 更新统计信息
4. 防止事务 ID 回卷

**类型：**

- `VACUUM`：标记空间可重用，不释放给操作系统
- `VACUUM FULL`：完全释放空间，但会锁表
- `VACUUM ANALYZE`：同时更新统计信息

```sql
VACUUM users;
VACUUM ANALYZE users;
```

## 🚀 高级题

### 8. 如何优化一个慢查询？请给出步骤。

**答案：**

**优化步骤：**

1. **使用 EXPLAIN ANALYZE 分析**

```sql
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 123;
```

2. **查看执行计划**

- Seq Scan（全表扫描）→ 需要优化
- Index Scan（索引扫描）→ 较好
- cost、actual time 指标

3. **创建索引**

```sql
CREATE INDEX idx_orders_user_id ON orders(user_id);
```

4. **更新统计信息**

```sql
ANALYZE orders;
```

5. **优化查询语句**

- 避免 SELECT \*
- 使用 JOIN 代替子查询
- 合理使用 WHERE 条件

6. **考虑分区**

```sql
CREATE TABLE orders (...) PARTITION BY RANGE (created_at);
```

### 9. 什么是死锁？如何避免？

**答案：**

**死锁定义：**
两个或多个事务相互等待对方释放锁，导致无法继续执行。

**示例：**

```sql
-- 事务 1
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- 事务 2（同时执行）
BEGIN;
UPDATE accounts SET balance = balance - 50 WHERE id = 2;
UPDATE accounts SET balance = balance + 50 WHERE id = 1;
COMMIT;
```

**避免方法：**

1. **按相同顺序访问资源**
2. **缩短事务时间**
3. **使用较低的隔离级别**
4. **使用 SKIP LOCKED**

```sql
SELECT * FROM jobs
WHERE status = 'pending'
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

### 10. 如何实现主从复制？

**答案：**

**1. 主库配置（postgresql.conf）：**

```conf
wal_level = replica
max_wal_senders = 3
wal_keep_size = 64
```

**2. 创建复制用户：**

```sql
CREATE USER replicator REPLICATION LOGIN PASSWORD 'password';
```

**3. 配置 pg_hba.conf：**

```conf
host replication replicator slave_ip/32 md5
```

**4. 从库初始化：**

```bash
pg_basebackup -h master_ip -D /var/lib/postgresql/data -U replicator -P -Xs -R
```

**5. 启动从库：**

```bash
sudo systemctl start postgresql
```

### 11. 分区表的优缺点？

**答案：**

**优点：**

1. **查询性能提升**：只扫描相关分区
2. **维护方便**：可单独备份/删除分区
3. **并行处理**：不同分区可并行查询
4. **数据管理**：旧数据可归档到独立分区

**缺点：**

1. **增加复杂性**：需要设计分区策略
2. **约束限制**：主键和唯一约束必须包含分区键
3. **跨分区查询**：可能影响性能

**示例：**

```sql
CREATE TABLE orders (
    id SERIAL,
    created_at DATE NOT NULL,
    total NUMERIC(10, 2)
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2024_q1 PARTITION OF orders
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
```

### 12. 如何实现数据库的高可用？

**答案：**

**常见方案：**

1. **主从复制 + 故障转移**

- 使用 Streaming Replication
- 配合 Patroni/Repmgr 自动故障转移

2. **Patroni + etcd/Consul**

- 自动故障检测和切换
- 健康检查
- 负载均衡

3. **PgPool-II**

- 连接池
- 负载均衡
- 故障转移

4. **云服务**

- AWS RDS
- Google Cloud SQL
- Azure Database for PostgreSQL

## 💡 实战题

### 13. 设计一个转账系统的数据库表结构

**答案：**

```sql
-- 账户表
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    balance NUMERIC(15, 2) NOT NULL CHECK (balance >= 0),
    currency VARCHAR(3) DEFAULT 'CNY',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 转账记录表
CREATE TABLE transfers (
    id SERIAL PRIMARY KEY,
    from_account_id INT NOT NULL,
    to_account_id INT NOT NULL,
    amount NUMERIC(15, 2) NOT NULL CHECK (amount > 0),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    FOREIGN KEY (from_account_id) REFERENCES accounts(id),
    FOREIGN KEY (to_account_id) REFERENCES accounts(id)
);

-- 转账函数
CREATE OR REPLACE FUNCTION transfer(
    p_from_account INT,
    p_to_account INT,
    p_amount NUMERIC
) RETURNS INT AS $$
DECLARE
    v_transfer_id INT;
BEGIN
    -- 检查余额
    IF (SELECT balance FROM accounts WHERE id = p_from_account) < p_amount THEN
        RAISE EXCEPTION 'Insufficient funds';
    END IF;

    -- 插入转账记录
    INSERT INTO transfers (from_account_id, to_account_id, amount, status)
    VALUES (p_from_account, p_to_account, p_amount, 'processing')
    RETURNING id INTO v_transfer_id;

    -- 扣款
    UPDATE accounts SET balance = balance - p_amount WHERE id = p_from_account;

    -- 入账
    UPDATE accounts SET balance = balance + p_amount WHERE id = p_to_account;

    -- 更新转账状态
    UPDATE transfers
    SET status = 'completed', completed_at = NOW()
    WHERE id = v_transfer_id;

    RETURN v_transfer_id;
END;
$$ LANGUAGE plpgsql;
```

### 14. 如何处理大量数据的分页查询？

**答案：**

**问题：** 使用 `LIMIT + OFFSET` 在大偏移量时性能很差。

```sql
-- ❌ 性能差（大偏移量）
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 100000;
```

**解决方案：**

1. **使用游标分页（推荐）**

```sql
SELECT * FROM users
WHERE id > 100000  -- 上一页最后一个 ID
ORDER BY id
LIMIT 10;
```

2. **使用 keyset 分页**

```sql
SELECT * FROM users
WHERE (created_at, id) > ('2024-01-01 00:00:00', 12345)
ORDER BY created_at, id
LIMIT 10;
```

3. **使用物化视图**

```sql
CREATE MATERIALIZED VIEW user_pages AS
SELECT id, username, email, ROW_NUMBER() OVER (ORDER BY id) as row_num
FROM users;
```

## 📚 相关资源

- [基础概念](/docs/postgres/basic-concepts)
- [性能优化](/docs/postgres/performance-optimization)
- [事务管理](/docs/postgres/transactions)
