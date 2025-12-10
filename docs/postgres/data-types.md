---
sidebar_position: 3
title: 数据类型
---

# PostgreSQL 数据类型

PostgreSQL 支持丰富的数据类型，包括基本类型、高级类型和自定义类型。

## 📝 数值类型

### 整数类型

| 类型        | 存储大小 | 范围                                        | 说明       |
| ----------- | -------- | ------------------------------------------- | ---------- |
| `SMALLINT`  | 2 字节   | -32768 到 32767                             | 小整数     |
| `INTEGER`   | 4 字节   | -2147483648 到 2147483647                   | 常用整数   |
| `BIGINT`    | 8 字节   | -9223372036854775808 到 9223372036854775807 | 大整数     |
| `SERIAL`    | 4 字节   | 1 到 2147483647                             | 自增整数   |
| `BIGSERIAL` | 8 字节   | 1 到 9223372036854775807                    | 大自增整数 |

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    stock_quantity SMALLINT,
    total_sales BIGINT
);
```

### 浮点类型

| 类型               | 存储大小 | 精度                             | 说明           |
| ------------------ | -------- | -------------------------------- | -------------- |
| `REAL`             | 4 字节   | 6 位小数                         | 单精度浮点数   |
| `DOUBLE PRECISION` | 8 字节   | 15 位小数                        | 双精度浮点数   |
| `NUMERIC(p,s)`     | 可变     | 最多 131072 位整数，16383 位小数 | 精确数值       |
| `DECIMAL(p,s)`     | 可变     | 同 NUMERIC                       | NUMERIC 的别名 |

```sql
CREATE TABLE financial_data (
    price NUMERIC(10, 2),      -- 10 位数字，2 位小数
    discount REAL,
    exchange_rate DOUBLE PRECISION
);
```

## 📅 日期/时间类型

| 类型          | 存储大小 | 范围                         | 说明           |
| ------------- | -------- | ---------------------------- | -------------- |
| `DATE`        | 4 字节   | 4713 BC 到 5874897 AD        | 日期           |
| `TIME`        | 8 字节   | 00:00:00 到 24:00:00         | 时间           |
| `TIMESTAMP`   | 8 字节   | 4713 BC 到 294276 AD         | 日期和时间     |
| `TIMESTAMPTZ` | 8 字节   | 同上                         | 带时区的时间戳 |
| `INTERVAL`    | 16 字节  | -178000000 年到 178000000 年 | 时间间隔       |

```sql
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    event_date DATE,
    event_time TIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 日期时间操作
SELECT
    CURRENT_DATE,                          -- 当前日期
    CURRENT_TIME,                          -- 当前时间
    NOW(),                                 -- 当前时间戳
    CURRENT_DATE + INTERVAL '1 day',       -- 明天
    AGE(TIMESTAMP '2024-01-01', TIMESTAMP '2023-01-01');  -- 时间差
```

## 🔤 字符类型

| 类型         | 说明                      |
| ------------ | ------------------------- |
| `CHAR(n)`    | 定长字符串，不足补空格    |
| `VARCHAR(n)` | 变长字符串，最多 n 个字符 |
| `TEXT`       | 无限长度的变长字符串      |

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    country_code CHAR(2),        -- 固定 2 字符，如 'CN'
    username VARCHAR(50),         -- 最多 50 字符
    bio TEXT                      -- 无限制
);

-- 推荐使用 VARCHAR 或 TEXT
-- PostgreSQL 对 TEXT 进行了优化，性能与 VARCHAR 相当
```

## ✅ 布尔类型

```sql
CREATE TABLE settings (
    id SERIAL PRIMARY KEY,
    is_enabled BOOLEAN DEFAULT true,
    is_admin BOOLEAN NOT NULL
);

-- 布尔值
INSERT INTO settings (is_enabled, is_admin)
VALUES (true, false);

-- 可以使用字符串
INSERT INTO settings (is_enabled, is_admin)
VALUES ('yes', 'no');  -- 'yes', 'on', '1', 'true' 都是 true

-- 查询
SELECT * FROM settings WHERE is_enabled = true;
SELECT * FROM settings WHERE is_enabled;  -- 简写
```

## 📦 数组类型

PostgreSQL 支持任意数据类型的数组。

```sql
CREATE TABLE blog_posts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    tags TEXT[],                 -- 文本数组
    ratings INTEGER[]            -- 整数数组
);

-- 插入数组
INSERT INTO blog_posts (title, tags, ratings)
VALUES (
    'PostgreSQL Tutorial',
    ARRAY['database', 'sql', 'postgresql'],
    ARRAY[5, 4, 5, 5]
);

-- 或使用花括号语法
INSERT INTO blog_posts (title, tags)
VALUES ('Advanced SQL', '{"sql", "advanced", "tutorial"}');

-- 查询数组
SELECT * FROM blog_posts WHERE 'sql' = ANY(tags);
SELECT * FROM blog_posts WHERE tags @> ARRAY['database'];  -- 包含
SELECT * FROM blog_posts WHERE tags && ARRAY['sql', 'java'];  -- 重叠

-- 数组函数
SELECT
    array_length(tags, 1) as tag_count,  -- 数组长度
    tags[1] as first_tag,                 -- 第一个元素（从 1 开始）
    array_append(tags, 'new') as new_tags -- 追加元素
FROM blog_posts;
```

## 📋 JSON 类型

PostgreSQL 支持两种 JSON 类型：`JSON` 和 `JSONB`。

| 类型    | 存储       | 性能   | 索引   | 推荐 |
| ------- | ---------- | ------ | ------ | ---- |
| `JSON`  | 文本格式   | 输入快 | 不支持 | ❌   |
| `JSONB` | 二进制格式 | 查询快 | 支持   | ✅   |

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    metadata JSONB
);

-- 插入 JSON 数据
INSERT INTO users (name, metadata) VALUES
('Alice', '{"age": 30, "city": "Beijing", "hobbies": ["reading", "coding"]}'),
('Bob', '{"age": 25, "city": "Shanghai"}');

-- 查询 JSON 字段
SELECT
    name,
    metadata->>'age' as age,           -- 提取为文本
    metadata->'hobbies' as hobbies,    -- 提取为 JSON
    metadata->'hobbies'->0 as first_hobby  -- 数组第一个元素
FROM users;

-- JSON 查询操作符
SELECT * FROM users WHERE metadata->>'city' = 'Beijing';
SELECT * FROM users WHERE metadata @> '{"age": 30}';  -- 包含
SELECT * FROM users WHERE metadata ? 'age';            -- 存在键

-- JSON 函数
SELECT
    jsonb_object_keys(metadata) as keys,       -- 所有键
    jsonb_pretty(metadata) as formatted         -- 格式化输出
FROM users;

-- 创建 GIN 索引加速查询
CREATE INDEX idx_users_metadata ON users USING GIN (metadata);
```

## 🆔 UUID 类型

通用唯一标识符，适合作为分布式系统的主键。

```sql
-- 启用 UUID 扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_number VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 插入数据
INSERT INTO orders (order_number) VALUES ('ORD-001');

-- 查询
SELECT * FROM orders WHERE id = '550e8400-e29b-41d4-a716-446655440000';
```

## 📐 几何类型

用于存储几何数据。

```sql
CREATE TABLE locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    coordinates POINT,      -- 点
    area POLYGON            -- 多边形
);

-- 插入数据
INSERT INTO locations (name, coordinates)
VALUES ('Office', POINT(116.4074, 39.9042));  -- 北京坐标

-- 查询距离
SELECT name, coordinates <-> POINT(0, 0) as distance
FROM locations
ORDER BY distance;
```

## 🔗 枚举类型

自定义枚举类型。

```sql
-- 创建枚举类型
CREATE TYPE status_type AS ENUM ('pending', 'processing', 'completed', 'cancelled');

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_number VARCHAR(50),
    status status_type DEFAULT 'pending'
);

-- 插入数据
INSERT INTO orders (order_number, status)
VALUES ('ORD-001', 'processing');

-- 查询
SELECT * FROM orders WHERE status = 'completed';

-- 修改枚举（添加新值）
ALTER TYPE status_type ADD VALUE 'refunded';
```

## 📊 范围类型

表示值的范围。

```sql
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    date_range DATERANGE,       -- 日期范围
    price_range INT4RANGE       -- 整数范围
);

-- 插入数据
INSERT INTO events (name, date_range, price_range) VALUES
('Conference', '[2024-06-01, 2024-06-03]', '[100, 500)');

-- 范围查询
SELECT * FROM events
WHERE date_range @> '2024-06-02'::DATE;  -- 包含日期

SELECT * FROM events
WHERE date_range && '[2024-06-01, 2024-06-05]'::DATERANGE;  -- 重叠
```

## 💡 类型转换

```sql
-- 显式转换
SELECT CAST('123' AS INTEGER);
SELECT '123'::INTEGER;
SELECT TO_NUMBER('123.45', '999.99');

-- 日期转换
SELECT TO_DATE('2024-01-15', 'YYYY-MM-DD');
SELECT TO_TIMESTAMP('2024-01-15 14:30:00', 'YYYY-MM-DD HH24:MI:SS');

-- 文本转换
SELECT TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS');
SELECT TO_CHAR(1234.5, '9999.99');
```

## 🎯 选择建议

| 场景   | 推荐类型                       | 说明                |
| ------ | ------------------------------ | ------------------- |
| 主键   | `SERIAL`/`BIGSERIAL` 或 `UUID` | 自增 ID 或分布式 ID |
| 金额   | `NUMERIC(10,2)`                | 避免浮点误差        |
| 文本   | `VARCHAR` 或 `TEXT`            | TEXT 性能已优化     |
| 时间戳 | `TIMESTAMPTZ`                  | 自动处理时区        |
| JSON   | `JSONB`                        | 支持索引和高效查询  |
| 布尔   | `BOOLEAN`                      | 明确的真/假值       |

## 📚 相关资源

- [基础概念](./basic-concepts) - 了解数据库基础
- [SQL 语法](./sql-syntax) - 学习 SQL 查询
- [索引优化](./indexes) - 优化查询性能

下一节：[SQL 语法](./sql-syntax)
