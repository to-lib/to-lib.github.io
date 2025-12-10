---
sidebar_position: 4
title: SQL 语法详解
---

# SQL 语法详解

> [!TIP] > **学习重点**: SQL 是操作数据库的核心语言。掌握 DDL、DML 和查询语句是数据库开发的基础技能。

## SQL 语句分类

SQL 语句主要分为以下几类：

- **DDL** (Data Definition Language) - 数据定义语言：CREATE、ALTER、DROP
- **DML** (Data Manipulation Language) - 数据操作语言：INSERT、UPDATE、DELETE、SELECT
- **DCL** (Data Control Language) - 数据控制语言：GRANT、REVOKE
- **TCL** (Transaction Control Language) - 事务控制语言：COMMIT、ROLLBACK

## DDL - 数据定义语言

### CREATE - 创建

#### 创建数据库

```sql
-- 基本语法
CREATE DATABASE database_name;

-- 指定字符集
CREATE DATABASE mydb
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

-- 如果不存在再创建
CREATE DATABASE IF NOT EXISTS mydb;
```

#### 创建表

```sql
-- 基本建表
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 完整建表示例
CREATE TABLE IF NOT EXISTS orders (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '订单ID',
    order_no VARCHAR(32) NOT NULL COMMENT '订单号',
    user_id BIGINT UNSIGNED NOT NULL COMMENT '用户ID',
    total_amount DECIMAL(10,2) NOT NULL DEFAULT 0.00 COMMENT '总金额',
    status TINYINT NOT NULL DEFAULT 0 COMMENT '状态: 0-待支付 1-已支付 2-已取消',
    paid_at TIMESTAMP NULL COMMENT '支付时间',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                  ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE KEY uk_order_no (order_no),
    KEY idx_user_id (user_id),
    KEY idx_status (status),
    KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='订单表';
```

#### 从查询结果创建表

```sql
-- 复制表结构和数据
CREATE TABLE users_backup AS SELECT * FROM users;

-- 只复制表结构
CREATE TABLE users_empty LIKE users;

-- 复制部分数据
CREATE TABLE vip_users AS
SELECT * FROM users WHERE vip_level > 0;
```

### ALTER - 修改表结构

#### 添加列

```sql
-- 添加单列
ALTER TABLE users ADD COLUMN age TINYINT;

-- 添加列并指定位置
ALTER TABLE users ADD COLUMN gender TINYINT AFTER username;
ALTER TABLE users ADD COLUMN nickname VARCHAR(50) FIRST;

-- 添加多列
ALTER TABLE users
ADD COLUMN phone CHAR(11),
ADD COLUMN address VARCHAR(200);
```

#### 修改列

```sql
-- 修改列数据类型
ALTER TABLE users MODIFY COLUMN age SMALLINT;

-- 修改列名和数据类型
ALTER TABLE users CHANGE COLUMN age user_age TINYINT;

-- 修改列默认值
ALTER TABLE users ALTER COLUMN status SET DEFAULT 1;
ALTER TABLE users ALTER COLUMN status DROP DEFAULT;
```

#### 删除列

```sql
-- 删除单列
ALTER TABLE users DROP COLUMN age;

-- 删除多列
ALTER TABLE users
DROP COLUMN age,
DROP COLUMN gender;
```

#### 表重命名

```sql
-- 方式1
ALTER TABLE users RENAME TO members;

-- 方式2
RENAME TABLE members TO users;

-- 批量重命名
RENAME TABLE
    old_table1 TO new_table1,
    old_table2 TO new_table2;
```

### DROP - 删除

```sql
-- 删除数据库
DROP DATABASE mydb;
DROP DATABASE IF EXISTS mydb;

-- 删除表
DROP TABLE users;
DROP TABLE IF EXISTS users;

-- 删除多张表
DROP TABLE IF EXISTS table1, table2, table3;
```

### TRUNCATE - 清空表

```sql
-- 清空表（保留结构，删除所有数据）
TRUNCATE TABLE users;

-- TRUNCATE vs DELETE
-- TRUNCATE: 快速清空，重置自增ID，不能回滚
-- DELETE: 逐行删除，不重置自增ID，可以回滚
```

## DML - 数据操作语言

### INSERT - 插入数据

#### 基本插入

```sql
-- 插入完整行
INSERT INTO users (username, email, age)
VALUES ('张三', 'zhangsan@example.com', 25);

-- 插入多行
INSERT INTO users (username, email, age) VALUES
('李四', 'lisi@example.com', 30),
('王五', 'wangwu@example.com', 28),
('赵六', 'zhaoliu@example.com', 35);

-- 省略列名（按表结构顺序）
INSERT INTO users VALUES (NULL, '小明', 'xiaoming@example.com', 20, NOW());
```

#### 特殊插入

```sql
-- 插入或忽略（如果主键/唯一键冲突则忽略）
INSERT IGNORE INTO users (id, username) VALUES (1, '张三');

-- 插入或替换（如果冲突则替换）
REPLACE INTO users (id, username, email) VALUES (1, '张三', 'new@example.com');

-- 插入或更新（ON DUPLICATE KEY UPDATE）
INSERT INTO users (id, username, email)
VALUES (1, '张三', 'zhangsan@example.com')
ON DUPLICATE KEY UPDATE
    email = VALUES(email),
    updated_at = NOW();
```

#### 从查询结果插入

```sql
-- 从其他表插入数据
INSERT INTO users_backup (username, email)
SELECT username, email FROM users WHERE age > 18;
```

### UPDATE - 更新数据

#### 基本更新

```sql
-- 更新单个字段
UPDATE users SET age = 26 WHERE id = 1;

-- 更新多个字段
UPDATE users
SET age = 26, email = 'newemail@example.com'
WHERE id = 1;

-- 更新所有行（危险操作！）
UPDATE users SET status = 1;
```

#### 高级更新

```sql
-- 基于计算更新
UPDATE products SET price = price * 0.9 WHERE category = 'electronics';

-- 使用 CASE 条件更新
UPDATE users
SET level = CASE
    WHEN points >= 1000 THEN 'VIP'
    WHEN points >= 500 THEN 'Gold'
    ELSE 'Silver'
END;

-- 多表更新
UPDATE users u
INNER JOIN orders o ON u.id = o.user_id
SET u.total_orders = u.total_orders + 1
WHERE o.status = 'completed';

-- 限制更新行数
UPDATE users SET status = 1 WHERE age > 18 LIMIT 100;
```

### DELETE - 删除数据

#### 基本删除

```sql
-- 删除特定行
DELETE FROM users WHERE id = 1;

-- 删除多行
DELETE FROM users WHERE age < 18;

-- 删除所有行（保留表结构）
DELETE FROM users;
```

#### 高级删除

```sql
-- 限制删除行数
DELETE FROM users WHERE status = 0 LIMIT 100;

-- 多表删除
DELETE u, o
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE u.status = 'inactive';

-- 根据其他表条件删除
DELETE FROM users
WHERE id IN (SELECT user_id FROM banned_users);
```

> [!WARNING] > **安全提醒**:
>
> - UPDATE 和 DELETE 操作前务必先用 SELECT 验证条件
> - 生产环境建议添加 LIMIT 限制
> - 重要操作前先备份数据

### SELECT - 查询数据

#### 基本查询

```sql
-- 查询所有列
SELECT * FROM users;

-- 查询指定列
SELECT id, username, email FROM users;

-- 列别名
SELECT
    id AS user_id,
    username AS name,
    email AS contact_email
FROM users;

-- DISTINCT 去重
SELECT DISTINCT city FROM users;

-- LIMIT 限制结果数量
SELECT * FROM users LIMIT 10;
SELECT * FROM users LIMIT 10, 20;  -- 跳过10条，取20条
SELECT * FROM users LIMIT 20 OFFSET 10;  -- 等同于上面
```

#### WHERE 条件查询

```sql
-- 比较运算符
SELECT * FROM users WHERE age > 18;
SELECT * FROM users WHERE age >= 18 AND age <= 30;
SELECT * FROM users WHERE age BETWEEN 18 AND 30;

-- IN 和 NOT IN
SELECT * FROM users WHERE city IN ('北京', '上海', '深圳');
SELECT * FROM users WHERE city NOT IN ('北京', '上海');

-- LIKE 模糊查询
SELECT * FROM users WHERE username LIKE '张%';     -- 张开头
SELECT * FROM users WHERE username LIKE '%三%';    -- 包含三
SELECT * FROM users WHERE email LIKE '%@gmail.com';  -- gmail邮箱

-- IS NULL 和 IS NOT NULL
SELECT * FROM users WHERE phone IS NULL;
SELECT * FROM users WHERE phone IS NOT NULL;

-- 逻辑运算符
SELECT * FROM users
WHERE age > 18 AND (city = '北京' OR city = '上海');
```

#### ORDER BY 排序

```sql
-- 升序（默认）
SELECT * FROM users ORDER BY age;
SELECT * FROM users ORDER BY age ASC;

-- 降序
SELECT * FROM users ORDER BY created_at DESC;

-- 多字段排序
SELECT * FROM users
ORDER BY age DESC, username ASC;

-- 按表达式排序
SELECT * FROM products
ORDER BY price * discount_rate DESC;
```

#### GROUP BY 分组

```sql
-- 基本分组
SELECT city, COUNT(*) as user_count
FROM users
GROUP BY city;

-- 多字段分组
SELECT city, gender, COUNT(*) as count
FROM users
GROUP BY city, gender;

-- HAVING 过滤分组
SELECT city, COUNT(*) as user_count
FROM users
GROUP BY city
HAVING user_count > 100;

-- 分组聚合函数
SELECT
    city,
    COUNT(*) as total,
    AVG(age) as avg_age,
    MAX(age) as max_age,
    MIN(age) as min_age,
    SUM(balance) as total_balance
FROM users
GROUP BY city;
```

#### JOIN 连接查询

##### INNER JOIN（内连接）

```sql
-- 基本内连接
SELECT u.username, o.order_no, o.total_amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- 多表连接
SELECT
    u.username,
    o.order_no,
    oi.product_name,
    oi.quantity
FROM users u
INNER JOIN orders o ON u.id = o.user_id
INNER JOIN order_items oi ON o.id = oi.order_id;
```

##### LEFT JOIN（左连接）

```sql
-- 查询所有用户及其订单（包括没有订单的用户）
SELECT u.username, o.order_no
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- 查找没有订单的用户
SELECT u.username
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;
```

##### RIGHT JOIN（右连接）

```sql
-- 查询所有订单及对应用户
SELECT u.username, o.order_no
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;
```

##### CROSS JOIN（交叉连接）

```sql
-- 笛卡尔积
SELECT u.username, p.product_name
FROM users u
CROSS JOIN products p;
```

#### 子查询

##### WHERE 子查询

```sql
-- IN 子查询
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE status = 'paid');

-- 比较子查询
SELECT * FROM products
WHERE price > (SELECT AVG(price) FROM products);

-- EXISTS 子查询
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.user_id = u.id AND o.status = 'paid'
);
```

##### FROM 子查询

```sql
-- 在 FROM 中使用子查询
SELECT city, avg_age, user_count
FROM (
    SELECT city, AVG(age) as avg_age, COUNT(*) as user_count
    FROM users
    GROUP BY city
) AS city_stats
WHERE avg_age > 25;
```

##### SELECT 子查询

```sql
-- 在 SELECT 中使用子查询
SELECT
    id,
    username,
    (SELECT COUNT(*) FROM orders WHERE user_id = users.id) as order_count
FROM users;
```

#### UNION 联合查询

```sql
-- UNION（去重）
SELECT username FROM users WHERE city = '北京'
UNION
SELECT username FROM users WHERE age > 30;

-- UNION ALL（不去重，性能更好）
SELECT username FROM users WHERE city = '北京'
UNION ALL
SELECT username FROM users WHERE city = '上海';
```

## 聚合函数

### 常用聚合函数

```sql
-- COUNT：计数
SELECT COUNT(*) FROM users;                    -- 总行数
SELECT COUNT(phone) FROM users;                -- phone 不为NULL的行数
SELECT COUNT(DISTINCT city) FROM users;        -- 不同城市数量

-- SUM：求和
SELECT SUM(total_amount) FROM orders;

-- AVG：平均值
SELECT AVG(age) FROM users;
SELECT AVG(price) FROM products WHERE category = 'electronics';

-- MAX/MIN：最大值/最小值
SELECT MAX(age), MIN(age) FROM users;

-- 组合使用
SELECT
    COUNT(*) as total_users,
    AVG(age) as avg_age,
    MAX(balance) as max_balance,
    SUM(balance) as total_balance
FROM users;
```

## 常用函数

### 字符串函数

```sql
-- CONCAT：拼接字符串
SELECT CONCAT(username, '@', email) as full_info FROM users;

-- SUBSTRING：截取字符串
SELECT SUBSTRING(username, 1, 3) FROM users;

-- LENGTH：字符串长度
SELECT username, LENGTH(username) as name_length FROM users;

-- UPPER/LOWER：大小写转换
SELECT UPPER(username), LOWER(email) FROM users;

-- TRIM：去除空格
SELECT TRIM(username) FROM users;

-- REPLACE：替换
SELECT REPLACE(email, '@gmail.com', '@example.com') FROM users;
```

### 数值函数

```sql
-- ROUND：四舍五入
SELECT ROUND(price, 2) FROM products;

-- CEIL：向上取整
SELECT CEIL(3.14);  -- 4

-- FLOOR：向下取整
SELECT FLOOR(3.99);  -- 3

-- ABS：绝对值
SELECT ABS(-10);  -- 10

-- RAND：随机数
SELECT * FROM products ORDER BY RAND() LIMIT 10;
```

### 日期时间函数

```sql
-- NOW：当前日期时间
SELECT NOW();  -- 2025-12-09 22:00:00

-- CURDATE：当前日期
SELECT CURDATE();  -- 2025-12-09

-- CURTIME：当前时间
SELECT CURTIME();  -- 22:00:00

-- DATE：提取日期
SELECT DATE(created_at) FROM users;

-- YEAR/MONTH/DAY：提取年月日
SELECT YEAR(created_at), MONTH(created_at), DAY(created_at) FROM users;

-- DATE_FORMAT：格式化日期
SELECT DATE_FORMAT(created_at, '%Y年%m月%d日') FROM users;

-- DATE_ADD/DATE_SUB：日期计算
SELECT DATE_ADD(NOW(), INTERVAL 7 DAY);    -- 7天后
SELECT DATE_SUB(NOW(), INTERVAL 1 MONTH);  -- 1个月前

-- DATEDIFF：日期差
SELECT DATEDIFF(NOW(), created_at) as days_since_created FROM users;
```

### 条件函数

```sql
-- IF：条件判断
SELECT username, IF(age >= 18, '成年', '未成年') as age_group FROM users;

-- CASE WHEN：多条件判断
SELECT
    username,
    CASE
        WHEN age < 18 THEN '未成年'
        WHEN age BETWEEN 18 AND 60 THEN '成年'
        ELSE '老年'
    END as age_group
FROM users;

-- IFNULL：NULL 处理
SELECT username, IFNULL(phone, '未填写') as phone FROM users;

-- COALESCE：返回第一个非NULL值
SELECT COALESCE(phone, email, '无联系方式') as contact FROM users;
```

## 最佳实践

> [!IMPORTANT] > **SQL 编写规范**:
>
> 1. 使用大写关键字，提高可读性
> 2. 合理使用别名，简化 SQL
> 3. SELECT 避免使用 `*`，明确指定字段
> 4. WHERE 条件中索引列避免使用函数
> 5. JOIN 优于子查询（大多数情况）
> 6. 使用 EXPLAIN 分析查询性能

### 性能优化建议

```sql
-- ❌ 不推荐：全表扫描
SELECT * FROM users;

-- ✅ 推荐：指定字段和条件
SELECT id, username, email FROM users WHERE status = 1 LIMIT 100;

-- ❌ 不推荐：在索引列上使用函数
SELECT * FROM users WHERE YEAR(created_at) = 2025;

-- ✅ 推荐：避免函数，使用范围查询
SELECT * FROM users
WHERE created_at >= '2025-01-01' AND created_at < '2026-01-01';

-- ❌ 不推荐：OR 可能不走索引
SELECT * FROM users WHERE username = '张三' OR email = 'zhangsan@example.com';

-- ✅ 推荐：使用 UNION
SELECT * FROM users WHERE username = '张三'
UNION
SELECT * FROM users WHERE email = 'zhangsan@example.com';
```

## 总结

本文详细介绍了 MySQL 的 SQL 语法：

- ✅ DDL：CREATE、ALTER、DROP
- ✅ DML：INSERT、UPDATE、DELETE、SELECT
- ✅ 查询：WHERE、ORDER BY、GROUP BY、JOIN
- ✅ 子查询和 UNION
- ✅ 常用函数和最佳实践

掌握 SQL 语法后，可以继续学习 [索引优化](/docs/mysql/indexes) 和 [事务处理](/docs/mysql/transactions)！
