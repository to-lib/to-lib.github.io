---
sidebar_position: 5
title: 索引优化
---

# PostgreSQL 索引优化

索引是提升数据库查询性能的关键。合理使用索引可以大幅提升查询速度。

## 📖 索引基础

### 什么是索引？

索引是数据库表中一列或多列的值的排序副本，用于快速定位数据。

**类比**：索引就像书的目录，可以快速找到内容，而不用翻阅整本书。

### 何时使用索引？

✅ **应该创建索引：**

- WHERE 子句中频繁查询的列
- JOIN 条件中的列
- ORDER BY 和 GROUP BY 使用的列
- 外键列

❌ **不应创建索引：**

- 小表（全表扫描更快）
- 频繁更新的列
- 包含大量 NULL 值的列
- 返回大部分行的查询

## 🎯 索引类型

### 1. B-Tree 索引（默认）

最常用的索引类型，适用于大多数场景。

```sql
-- 创建 B-Tree 索引
CREATE INDEX idx_users_username ON users(username);

-- 等价于
CREATE INDEX idx_users_username ON users USING BTREE (username);

-- 适用场景
SELECT * FROM users WHERE username = 'john';
SELECT * FROM users WHERE username > 'a' AND username < 'z';
SELECT * FROM users ORDER BY username;
```

**特点：**

- 支持 `<, <=, =, >=, >` 操作符
- 支持 `BETWEEN, IN, IS NULL`
- 支持排序操作

### 2. Hash 索引

适用于等值查询。

```sql
CREATE INDEX idx_users_email ON users USING HASH (email);

-- 适用场景
SELECT * FROM users WHERE email = 'john@example.com';
```

**特点：**

- 只支持 `=` 操作符
- 比 B-Tree 稍快
- 不支持排序

### 3. GIN 索引（倒排索引）

适用于数组、JSON、全文搜索。

```sql
-- JSON 索引
CREATE INDEX idx_users_metadata ON users USING GIN (metadata);

-- 数组索引
CREATE INDEX idx_posts_tags ON posts USING GIN (tags);

-- 全文搜索索引
CREATE INDEX idx_articles_tsv ON articles USING GIN (tsv);

-- 查询
SELECT * FROM users WHERE metadata @> '{"city": "Beijing"}';
SELECT * FROM posts WHERE tags @> ARRAY['postgresql'];
```

**特点：**

- 适合多值列（数组、JSON）
- 全文搜索
- 创建慢，查询快
- 占用空间大

### 4. GiST 索引

适用于几何数据、全文搜索、范围类型。

```sql
-- 几何索引
CREATE INDEX idx_locations_coords ON locations USING GIST (coordinates);

-- 范围类型索引
CREATE INDEX idx_events_daterange ON events USING GIST (date_range);

-- 查询
SELECT * FROM events WHERE date_range @> '2024-01-15'::DATE;
```

### 5. BRIN 索引

适用于大表中具有自然顺序的列。

```sql
CREATE INDEX idx_logs_created_at ON logs USING BRIN (created_at);

-- 适用场景
SELECT * FROM logs WHERE created_at > '2024-01-01';
```

**特点：**

- 占用空间极小
- 适合时序数据
- 创建和维护快
- 查询性能中等

## 🔧 创建和管理索引

### 创建索引

```sql
-- 单列索引
CREATE INDEX idx_users_email ON users(email);

-- 多列索引（复合索引）
CREATE INDEX idx_users_name_age ON users(last_name, first_name, age);

-- 唯一索引
CREATE UNIQUE INDEX idx_users_username ON users(username);

-- 部分索引
CREATE INDEX idx_active_users ON users(username)
WHERE is_active = true;

-- 表达式索引
CREATE INDEX idx_users_lower_email ON users(LOWER(email));

-- 并发创建索引（不阻塞写操作）
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
```

### 删除索引

```sql
DROP INDEX idx_users_email;

-- 并发删除
DROP INDEX CONCURRENTLY idx_users_email;
```

### 重建索引

```sql
-- 重建单个索引
REINDEX INDEX idx_users_email;

-- 重建表的所有索引
REINDEX TABLE users;

-- 重建数据库的所有索引
REINDEX DATABASE myapp;

-- 并发重建
REINDEX INDEX CONCURRENTLY idx_users_email;
```

## 📊 复合索引

### 列顺序很重要

```sql
-- 索引：(last_name, first_name)
CREATE INDEX idx_users_name ON users(last_name, first_name);

-- ✅ 可以使用索引
SELECT * FROM users WHERE last_name = 'Smith';
SELECT * FROM users WHERE last_name = 'Smith' AND first_name = 'John';

-- ❌ 不能使用索引
SELECT * FROM users WHERE first_name = 'John';
```

**原则**：

1. 最左前缀原则
2. 选择性高的列放前面
3. 常用于等值查询的列放前面

### 覆盖索引

索引包含查询需要的所有列，无需回表。

```sql
-- 创建覆盖索引
CREATE INDEX idx_users_username_email ON users(username, email);

-- 直接从索引获取数据
SELECT username, email FROM users WHERE username = 'john';
```

## 🎓 部分索引

只对表的一部分创建索引，节省空间和提升性能。

```sql
-- 只索引活跃用户
CREATE INDEX idx_active_users ON users(username)
WHERE is_active = true;

-- 只索引最近的订单
CREATE INDEX idx_recent_orders ON orders(created_at)
WHERE created_at > '2024-01-01';

-- 查询必须包含相同条件
SELECT * FROM users
WHERE username = 'john' AND is_active = true;
```

## 🧪 表达式索引

对列的表达式创建索引。

```sql
-- 大小写不敏感搜索
CREATE INDEX idx_users_lower_email ON users(LOWER(email));

SELECT * FROM users WHERE LOWER(email) = 'john@example.com';

-- JSON 字段索引
CREATE INDEX idx_users_metadata_city ON users((metadata->>'city'));

SELECT * FROM users WHERE metadata->>'city' = 'Beijing';
```

## 📈 查询分析

### EXPLAIN - 查看查询计划

```sql
EXPLAIN SELECT * FROM users WHERE username = 'john';
```

输出示例：

```
Seq Scan on users  (cost=0.00..35.50 rows=10 width=524)
  Filter: (username = 'john'::text)
```

### EXPLAIN ANALYZE - 实际执行并分析

```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE username = 'john';
```

输出示例：

```
Index Scan using idx_users_username on users  (cost=0.15..8.17 rows=1 width=524)
  (actual time=0.025..0.026 rows=1 loops=1)
  Index Cond: (username = 'john'::text)
Planning Time: 0.123 ms
Execution Time: 0.045 ms
```

### 关键指标

- **Seq Scan**：全表扫描（慢）
- **Index Scan**：索引扫描（快）
- **cost**：估算成本
- **rows**：估算行数
- **actual time**：实际执行时间
- **loops**：执行次数

## 💡 优化技巧

### 1. 避免索引失效

```sql
-- ❌ 在索引列上使用函数
SELECT * FROM users WHERE UPPER(username) = 'JOHN';

-- ✅ 使用表达式索引
CREATE INDEX idx_users_upper_username ON users(UPPER(username));
SELECT * FROM users WHERE UPPER(username) = 'JOHN';

-- ❌ 使用 OR 连接不同列
SELECT * FROM users WHERE username = 'john' OR email = 'john@example.com';

-- ✅ 使用 UNION
SELECT * FROM users WHERE username = 'john'
UNION
SELECT * FROM users WHERE email = 'john@example.com';

-- ❌ 使用 NOT IN
SELECT * FROM users WHERE id NOT IN (1, 2, 3);

-- ✅ 使用 NOT EXISTS
SELECT * FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM blacklist b WHERE b.user_id = u.id
);
```

### 2. 选择合适的索引类型

```sql
-- 等值查询：B-Tree 或 Hash
CREATE INDEX idx_users_id ON users USING BTREE (id);

-- JSON 查询：GIN
CREATE INDEX idx_users_metadata ON users USING GIN (metadata);

-- 几何数据：GiST
CREATE INDEX idx_locations_coords ON locations USING GIST (coordinates);

-- 时序数据：BRIN
CREATE INDEX idx_logs_created_at ON logs USING BRIN (created_at);
```

### 3. 监控索引使用情况

```sql
-- 查看表的索引
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'users';

-- 查看索引大小
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
WHERE schemaname = 'public';

-- 查看未使用的索引
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND indexrelname NOT LIKE 'pg_toast%';
```

## ⚠️ 注意事项

1. **索引不是越多越好**

   - 占用存储空间
   - 影响写入性能
   - 需要维护成本

2. **定期维护索引**

   ```sql
   VACUUM ANALYZE users;  -- 更新统计信息
   REINDEX TABLE users;    -- 重建索引
   ```

3. **创建索引时加锁**
   ```sql
   -- 使用 CONCURRENTLY 避免锁表
   CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
   ```

## 📚 相关资源

- [SQL 语法](/docs/postgres/sql-syntax) - 学习 SQL 查询
- [性能优化](/docs/postgres/performance-optimization) - 全面优化指南
- [事务管理](/docs/postgres/transactions) - 了解事务

下一节：[事务管理](/docs/postgres/transactions)
