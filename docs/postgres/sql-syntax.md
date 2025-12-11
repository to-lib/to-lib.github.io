---
sidebar_position: 4
title: SQL 语法
---

# PostgreSQL SQL 语法

掌握 SQL 查询语言是使用 PostgreSQL 的基础。

## 📝 数据定义语言 (DDL)

### CREATE - 创建对象

```sql
-- 创建数据库
CREATE DATABASE myapp;

-- 创建表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    age INTEGER CHECK (age >= 0 AND age <= 150),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_users_username ON users(username);
CREATE UNIQUE INDEX idx_users_email ON users(email);

-- 创建视图
CREATE VIEW active_users AS
SELECT id, username, email
FROM users
WHERE created_at > NOW() - INTERVAL '30 days';
```

### ALTER - 修改对象

```sql
-- 添加列
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 删除列
ALTER TABLE users DROP COLUMN phone;

-- 修改列类型
ALTER TABLE users ALTER COLUMN age TYPE SMALLINT;

-- 重命名列
ALTER TABLE users RENAME COLUMN username TO user_name;

-- 添加约束
ALTER TABLE users ADD CONSTRAINT check_age CHECK (age >= 18);

-- 删除约束
ALTER TABLE users DROP CONSTRAINT check_age;
```

### DROP - 删除对象

```sql
-- 删除表
DROP TABLE users;

-- 如果存在则删除
DROP TABLE IF EXISTS users;

-- 级联删除（删除依赖对象）
DROP TABLE users CASCADE;
```

## 📊 数据操作语言 (DML)

### INSERT - 插入数据

```sql
-- 插入单行
INSERT INTO users (username, email, age)
VALUES ('john_doe', 'john@example.com', 25);

-- 插入多行
INSERT INTO users (username, email, age) VALUES
('alice', 'alice@example.com', 30),
('bob', 'bob@example.com', 28),
('charlie', 'charlie@example.com', 35);

-- 返回插入的数据
INSERT INTO users (username, email)
VALUES ('david', 'david@example.com')
RETURNING id, username, created_at;

-- 从查询结果插入
INSERT INTO archive_users
SELECT * FROM users WHERE created_at < '2020-01-01';

-- 冲突时不做任何操作
INSERT INTO users (username, email)
VALUES ('john_doe', 'john2@example.com')
ON CONFLICT (username) DO NOTHING;

-- 冲突时更新
INSERT INTO users (username, email, age)
VALUES ('john_doe', 'john@example.com', 26)
ON CONFLICT (username)
DO UPDATE SET email = EXCLUDED.email, age = EXCLUDED.age;
```

### SELECT - 查询数据

```sql
-- 基本查询
SELECT * FROM users;
SELECT username, email FROM users;

-- WHERE 条件
SELECT * FROM users WHERE age > 25;
SELECT * FROM users WHERE username = 'john_doe';
SELECT * FROM users WHERE age BETWEEN 20 AND 30;
SELECT * FROM users WHERE username IN ('alice', 'bob');
SELECT * FROM users WHERE email LIKE '%@example.com';
SELECT * FROM users WHERE username IS NOT NULL;

-- 排序
SELECT * FROM users ORDER BY age DESC;
SELECT * FROM users ORDER BY age ASC, username DESC;

-- 限制结果
SELECT * FROM users LIMIT 10;
SELECT * FROM users LIMIT 10 OFFSET 20;  -- 分页

-- 去重
SELECT DISTINCT age FROM users;

-- 聚合函数
SELECT COUNT(*) FROM users;
SELECT COUNT(DISTINCT age) FROM users;
SELECT AVG(age) as avg_age FROM users;
SELECT MAX(age) as max_age, MIN(age) as min_age FROM users;
SELECT SUM(age) FROM users;

-- 分组
SELECT age, COUNT(*) as count
FROM users
GROUP BY age;

SELECT age, COUNT(*) as count
FROM users
GROUP BY age
HAVING COUNT(*) > 1;

-- 子查询
SELECT * FROM users
WHERE age > (SELECT AVG(age) FROM users);

SELECT * FROM users
WHERE username IN (SELECT username FROM active_users);
```

### UPDATE - 更新数据

```sql
-- 更新单行
UPDATE users SET age = 26 WHERE username = 'john_doe';

-- 更新多列
UPDATE users
SET email = 'newemail@example.com', age = 27
WHERE username = 'john_doe';

-- 更新所有行
UPDATE users SET age = age + 1;

-- 返回更新的数据
UPDATE users SET age = 30
WHERE username = 'alice'
RETURNING id, username, age;

-- 基于子查询更新
UPDATE users
SET age = (SELECT AVG(age) FROM users)
WHERE age IS NULL;
```

### DELETE - 删除数据

```sql
-- 删除指定行
DELETE FROM users WHERE username = 'john_doe';

-- 删除所有行
DELETE FROM users;

-- 返回删除的数据
DELETE FROM users
WHERE age < 18
RETURNING id, username;

-- TRUNCATE - 快速清空表
TRUNCATE TABLE users;
TRUNCATE TABLE users RESTART IDENTITY;  -- 重置序列
```

## 🔗 连接查询 (JOIN)

### INNER JOIN

只返回两表中匹配的行。

```sql
SELECT users.username, orders.order_number
FROM users
INNER JOIN orders ON users.id = orders.user_id;
```

### LEFT JOIN

返回左表所有行，右表匹配的行，不匹配显示 NULL。

```sql
SELECT users.username, orders.order_number
FROM users
LEFT JOIN orders ON users.id = orders.user_id;
```

### RIGHT JOIN

返回右表所有行，左表匹配的行。

```sql
SELECT users.username, orders.order_number
FROM users
RIGHT JOIN orders ON users.id = orders.user_id;
```

### FULL OUTER JOIN

返回两表所有行，不匹配显示 NULL。

```sql
SELECT users.username, orders.order_number
FROM users
FULL OUTER JOIN orders ON users.id = orders.user_id;
```

### CROSS JOIN

笛卡尔积，返回两表所有组合。

```sql
SELECT users.username, products.name
FROM users
CROSS JOIN products;
```

### 多表连接

```sql
SELECT
    u.username,
    o.order_number,
    oi.quantity,
    p.name as product_name
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id;
```

## 🎯 高级查询

### WITH (公用表表达式 CTE)

```sql
-- 简单 CTE
WITH adult_users AS (
    SELECT * FROM users WHERE age >= 18
)
SELECT * FROM adult_users WHERE username LIKE 'a%';

-- 多个 CTE
WITH
    adult_users AS (
        SELECT * FROM users WHERE age >= 18
    ),
    active_users AS (
        SELECT * FROM users WHERE created_at > NOW() - INTERVAL '30 days'
    )
SELECT * FROM adult_users
INNER JOIN active_users ON adult_users.id = active_users.id;
```

### 递归 CTE

```sql
-- 组织层级结构
WITH RECURSIVE org_tree AS (
    -- 基础查询：顶层
    SELECT id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- 递归查询
    SELECT e.id, e.name, e.manager_id, ot.level + 1
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.id
)
SELECT * FROM org_tree;
```

### 窗口函数

```sql
-- ROW_NUMBER - 行号
SELECT
    username,
    age,
    ROW_NUMBER() OVER (ORDER BY age DESC) as rank
FROM users;

-- RANK / DENSE_RANK - 排名
SELECT
    username,
    age,
    RANK() OVER (ORDER BY age DESC) as rank,
    DENSE_RANK() OVER (ORDER BY age DESC) as dense_rank
FROM users;

-- PARTITION BY - 分组
SELECT
    department,
    username,
    salary,
    AVG(salary) OVER (PARTITION BY department) as dept_avg_salary
FROM employees;

-- LAG / LEAD - 前后行
SELECT
    date,
    revenue,
    LAG(revenue) OVER (ORDER BY date) as prev_revenue,
    LEAD(revenue) OVER (ORDER BY date) as next_revenue
FROM sales;
```

### CASE 表达式

```sql
-- 简单 CASE
SELECT
    username,
    age,
    CASE
        WHEN age < 18 THEN '未成年'
        WHEN age < 60 THEN '成年'
        ELSE '老年'
    END as age_group
FROM users;

-- 搜索 CASE
SELECT
    username,
    CASE username
        WHEN 'admin' THEN '管理员'
        WHEN 'guest' THEN '访客'
        ELSE '普通用户'
    END as user_type
FROM users;
```

## 🔍 全文搜索

```sql
-- 创建 tsvector 列
ALTER TABLE articles ADD COLUMN tsv tsvector;

-- 更新 tsvector
UPDATE articles
SET tsv = to_tsvector('english', title || ' ' || content);

-- 创建 GIN 索引
CREATE INDEX idx_articles_tsv ON articles USING GIN (tsv);

-- 全文搜索
SELECT title, content
FROM articles
WHERE tsv @@ to_tsquery('english', 'postgresql & database');

-- 相关性排序
SELECT
    title,
    ts_rank(tsv, query) as rank
FROM articles, to_tsquery('english', 'postgresql') query
WHERE tsv @@ query
ORDER BY rank DESC;
```

## 💡 最佳实践

1. **使用参数化查询**（防止 SQL 注入）
2. **避免 SELECT \***，明确指定列名
3. **合理使用索引**
4. **使用 EXPLAIN 分析查询**
5. **避免在 WHERE 中使用函数**

```sql
-- ❌ 不好
SELECT * FROM users WHERE LOWER(username) = 'john';

-- ✅ 好
SELECT id, username, email FROM users WHERE username = 'john';
CREATE INDEX idx_users_username_lower ON users(LOWER(username));
```

## 📚 相关资源

- [基础概念](/docs/postgres/basic-concepts) -了解数据库基础
- [数据类型](/docs/postgres/data-types) - 了解数据类型
- [索引优化](/docs/postgres/indexes) - 优化查询性能

下一节：[索引优化](/docs/postgres/indexes)
