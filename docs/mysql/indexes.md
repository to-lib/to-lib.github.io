---
sidebar_position: 5
title: 索引优化
---

# MySQL 索引优化

> [!TIP] > **性能关键**: 索引是提升 MySQL 查询性能的最重要手段。合理使用索引可以将查询速度提升数百倍甚至更多。

## 索引基础

### 什么是索引

索引是帮助 MySQL 高效获取数据的数据结构。类似于书的目录，可以快速定位到需要的内容。

### 索引的优缺点

#### 优点

- ✅ 大幅提升查询速度
- ✅ 加速表连接
- ✅ 减少分组和排序时间
- ✅ 唯一索引保证数据唯一性

#### 缺点

- ❌ 占用磁盘空间
- ❌ 降低写入速度（INSERT、UPDATE、DELETE）
- ❌ 维护成本（需要更新索引）

## 索引类型

### B-Tree 索引

最常用的索引类型，适合大多数场景。

#### 特点

- 数据有序存储
- 支持范围查询
- 支持排序
- 最左前缀原则

#### 使用示例

```sql
-- 创建单列索引
CREATE INDEX idx_username ON users(username);

-- 创建复合索引
CREATE INDEX idx_city_age ON users(city, age);

-- 唯一索引
CREATE UNIQUE INDEX uk_email ON users(email);

-- 查看索引
SHOW INDEX FROM users;
```

### Hash 索引

基于哈希表实现，仅 Memory 引擎支持（InnoDB 有自适应哈希索引）。

#### 特点

- ✅ 等值查询极快
- ❌ 不支持范围查询
- ❌ 不支持排序
- ❌ 不支持最左前缀

```sql
-- Memory 引擎使用 Hash 索引
CREATE TABLE cache_data (
    id INT PRIMARY KEY,
    key_name VARCHAR(50),
    value TEXT,
    INDEX USING HASH (key_name)
) ENGINE=Memory;
```

### Full-Text 索引

全文搜索索引，用于文本搜索。

```sql
-- 创建全文索引
CREATE FULLTEXT INDEX idx_content ON articles(title, content);

-- 使用全文搜索
SELECT * FROM articles
WHERE MATCH(title, content) AGAINST('MySQL 性能优化');

-- 布尔模式
SELECT * FROM articles
WHERE MATCH(title, content) AGAINST('+MySQL -Oracle' IN BOOLEAN MODE);
```

### 空间索引

用于地理空间数据（GEOMETRY 类型）。

```sql
CREATE TABLE locations (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    coordinates POINT NOT NULL,
    SPATIAL INDEX idx_coordinates (coordinates)
);
```

## 索引管理

### 创建索引

```sql
-- 方式1：CREATE INDEX
CREATE INDEX idx_username ON users(username);

-- 方式2：ALTER TABLE
ALTER TABLE users ADD INDEX idx_email (email);

-- 方式3：建表时创建
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    username VARCHAR(50),
    email VARCHAR(100),
    INDEX idx_username (username),
    UNIQUE INDEX uk_email (email)
);

-- 复合索引（多列索引）
CREATE INDEX idx_city_age_gender ON users(city, age, gender);

-- 前缀索引
CREATE INDEX idx_email_prefix ON users(email(10));
```

### 删除索引

```sql
-- 方式1：DROP INDEX
DROP INDEX idx_username ON users;

-- 方式2：ALTER TABLE
ALTER TABLE users DROP INDEX idx_username;
```

### 查看索引

```sql
-- 查看表的所有索引
SHOW INDEX FROM users;

-- 查看索引使用情况
SHOW INDEX FROM users\G

-- 查看索引统计信息
SELECT * FROM information_schema.STATISTICS
WHERE table_name = 'users';
```

## 索引设计原则

### 1. 在 WHERE 条件列上建索引

```sql
-- 经常查询的列
CREATE INDEX idx_status ON orders(status);

SELECT * FROM orders WHERE status = 'paid';  -- 使用索引
```

### 2. 在 JOIN 列上建索引

```sql
-- 外键列
CREATE INDEX idx_user_id ON orders(user_id);

SELECT * FROM orders o
INNER JOIN users u ON o.user_id = u.id;  -- 使用索引
```

### 3. 选择性高的列优先

```sql
-- ❌ 不适合：性别字段（选择性低）
CREATE INDEX idx_gender ON users(gender);  -- 只有2-3个值

-- ✅ 适合：用户名（选择性高）
CREATE INDEX idx_username ON users(username);  -- 每个用户唯一
```

### 4. 复合索引的列顺序

最左前缀原则：查询从索引的最左列开始，不能跳过中间列。

```sql
-- 创建复合索引
CREATE INDEX idx_city_age_gender ON users(city, age, gender);

-- ✅ 使用索引
SELECT * FROM users WHERE city = '北京';                          -- 使用 city
SELECT * FROM users WHERE city = '北京' AND age = 25;             -- 使用 city, age
SELECT * FROM users WHERE city = '北京' AND age = 25 AND gender = 1;  -- 使用全部

-- ❌ 不使用索引
SELECT * FROM users WHERE age = 25;                               -- 跳过 city
SELECT * FROM users WHERE gender = 1;                             -- 跳过 city, age
SELECT * FROM users WHERE age = 25 AND gender = 1;                -- 跳过 city
```

> [!IMPORTANT] > **复合索引列顺序规则**:
>
> 1. 区分度高的列放前面
> 2. 经常使用的列放前面
> 3. 范围查询的列放最后

```sql
-- 示例：优化后的顺序
CREATE INDEX idx_status_created ON orders(status, created_at);

-- ✅ 好：status 区分度高，且经常单独查询
SELECT * FROM orders WHERE status = 'paid';
SELECT * FROM orders WHERE status = 'paid' AND created_at > '2025-01-01';

-- ❌ 不好：如果把 created_at 放前面
-- CREATE INDEX idx_created_status ON orders(created_at, status);
-- 单独查询 status 将无法使用索引
```

### 5. 使用覆盖索引

覆盖索引：查询的列都在索引中，无需回表查询。

```sql
-- 创建覆盖索引
CREATE INDEX idx_username_email ON users(username, email);

-- ✅ 覆盖索引：无需回表
SELECT username, email FROM users WHERE username = '张三';

-- ❌ 非覆盖索引：需要回表获取 age
SELECT username, email, age FROM users WHERE username = '张三';
```

### 6. 避免冗余索引

```sql
-- ❌ 冗余：idx_username 被 idx_username_email 覆盖
CREATE INDEX idx_username ON users(username);
CREATE INDEX idx_username_email ON users(username, email);

-- ✅ 只保留复合索引
CREATE INDEX idx_username_email ON users(username, email);
```

## 索引失效场景

### 1. 在索引列上使用函数

```sql
-- ❌ 索引失效
SELECT * FROM users WHERE YEAR(created_at) = 2025;
SELECT * FROM users WHERE UPPER(username) = 'ZHANGSAN';

-- ✅ 改写查询
SELECT * FROM users
WHERE created_at >= '2025-01-01' AND created_at < '2026-01-01';
```

### 2. 使用 OR 条件

```sql
-- ❌ 可能不走索引
SELECT * FROM users WHERE username = '张三' OR age = 25;

-- ✅ 使用 UNION
SELECT * FROM users WHERE username = '张三'
UNION
SELECT * FROM users WHERE age = 25;

-- ✅ 或者两个列都有索引，且使用 IN
SELECT * FROM users WHERE id IN (1, 2, 3);
```

### 3. 类型不匹配

```sql
-- ❌ username 是 VARCHAR，使用数字查询
SELECT * FROM users WHERE username = 123;

-- ✅ 使用正确类型
SELECT * FROM users WHERE username = '123';
```

### 4. LIKE 以 % 开头

```sql
-- ❌ 索引失效
SELECT * FROM users WHERE username LIKE '%张三';

-- ✅ 索引有效
SELECT * FROM users WHERE username LIKE '张三%';
```

### 5. 使用 != 或 `<>`

```sql
-- ❌ 可能不走索引
SELECT * FROM users WHERE status != 1;

-- ✅ 改写为 IN 或范围查询
SELECT * FROM users WHERE status IN (0, 2, 3);
```

### 6. IS NOT NULL

```sql
-- ❌ 可能不走索引
SELECT * FROM users WHERE email IS NOT NULL;

-- ✅ 设计时避免 NULL，使用默认值
```

## EXPLAIN 执行计划分析

### 基本使用

```sql
EXPLAIN SELECT * FROM users WHERE username = '张三';
```

### 重要字段说明

| 字段          | 说明                                                                          |
| ------------- | ----------------------------------------------------------------------------- |
| id            | 查询序号                                                                      |
| select_type   | 查询类型（SIMPLE、PRIMARY、SUBQUERY 等）                                      |
| table         | 表名                                                                          |
| type          | 连接类型（性能从好到差：system > const > eq_ref > ref > range > index > ALL） |
| possible_keys | 可能使用的索引                                                                |
| key           | 实际使用的索引                                                                |
| key_len       | 索引使用的字节数                                                              |
| rows          | 扫描的行数                                                                    |
| Extra         | 额外信息                                                                      |

### type 类型详解

```sql
-- const：主键或唯一索引等值查询
EXPLAIN SELECT * FROM users WHERE id = 1;  -- type: const

-- ref：非唯一索引等值查询
EXPLAIN SELECT * FROM users WHERE username = '张三';  -- type: ref

-- range：范围查询
EXPLAIN SELECT * FROM users WHERE age BETWEEN 18 AND 30;  -- type: range

-- index：全索引扫描
EXPLAIN SELECT id FROM users;  -- type: index

-- ALL：全表扫描（最差）
EXPLAIN SELECT * FROM users WHERE age + 1 = 26;  -- type: ALL
```

### Extra 重要信息

```sql
-- Using index：覆盖索引
EXPLAIN SELECT username FROM users WHERE username = '张三';

-- Using where：使用 WHERE 过滤
EXPLAIN SELECT * FROM users WHERE age > 18;

-- Using filesort：文件排序（需要优化）
EXPLAIN SELECT * FROM users ORDER BY age;

-- Using temporary：使用临时表（需要优化）
EXPLAIN SELECT DISTINCT city FROM users;
```

## 索引优化案例

### 案例 1：慢查询优化

```sql
-- 原始查询（慢）
SELECT * FROM orders
WHERE user_id = 1000
AND status = 'paid'
AND created_at > '2025-01-01'
ORDER BY created_at DESC
LIMIT 10;

-- EXPLAIN 分析
EXPLAIN SELECT ...;  -- type: ALL, rows: 1000000

-- 优化：创建复合索引
CREATE INDEX idx_user_status_created ON orders(user_id, status, created_at);

-- 优化后
EXPLAIN SELECT ...;  -- type: ref, rows: 100
```

### 案例 2：覆盖索引优化

```sql
-- 原始查询
SELECT id, username, email FROM users WHERE username = '张三';

-- 创建覆盖索引
CREATE INDEX idx_username_email ON users(username, email);

-- EXPLAIN 显示 Using index（覆盖索引，性能最优）
```

### 案例 3：分页查询优化

```sql
-- ❌ 深分页慢查询
SELECT * FROM users ORDER BY id LIMIT 100000, 10;

-- ✅ 优化：使用子查询
SELECT * FROM users
WHERE id >= (SELECT id FROM users ORDER BY id LIMIT 100000, 1)
LIMIT 10;

-- ✅ 优化：使用上次最大 ID
SELECT * FROM users WHERE id > 100000 ORDER BY id LIMIT 10;
```

## 索引监控与维护

### 查看索引使用情况

```sql
-- 查看索引统计
SELECT
    TABLE_NAME,
    INDEX_NAME,
    SEQ_IN_INDEX,
    COLUMN_NAME,
    CARDINALITY
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = 'mydb';

-- 查看未使用的索引
SELECT * FROM sys.schema_unused_indexes;
```

### 索引维护

```sql
-- 重建索引
ALTER TABLE users DROP INDEX idx_username, ADD INDEX idx_username (username);

-- 分析表
ANALYZE TABLE users;

-- 优化表
OPTIMIZE TABLE users;
```

## 最佳实践总结

> [!IMPORTANT] > **索引设计黄金法则**:
>
> 1. 为 WHERE、JOIN、ORDER BY、GROUP BY 列创建索引
> 2. 选择性高的列优先索引
> 3. 使用复合索引，注意列顺序（最左前缀原则）
> 4. 利用覆盖索引避免回表
> 5. 避免索引失效场景（函数、OR、类型转换等）
> 6. 定期使用 EXPLAIN 分析查询
> 7. 删除冗余和未使用的索引

### 索引数量建议

- 单表索引数量不宜过多（一般不超过 5-6 个）
- 每个索引都有维护成本
- 定期检查并删除未使用的索引

### 何时不需要索引

- 表数据量很小（< 1000 行）
- 列的选择性很低（如性别、布尔值）
- 频繁更新的列
- 很少在 WHERE 中使用的列

## 总结

本文详细介绍了 MySQL 索引优化：

- ✅ 索引类型：B-Tree、Hash、Full-Text
- ✅ 索引创建和管理
- ✅ 索引设计原则（最左前缀、覆盖索引等）
- ✅ 索引失效场景及优化
- ✅ EXPLAIN 执行计划分析
- ✅ 实战优化案例

掌握索引优化后，可以继续学习 [事务处理](./transactions) 和 [性能优化](./performance-optimization)！
