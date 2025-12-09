---
sidebar_position: 3
title: MySQL 数据类型
---

# MySQL 数据类型

> [!TIP] > **性能优化关键**: 选择合适的数据类型可以显著提升数据库性能和节省存储空间。遵循"够用就好"的原则，不要浪费存储空间。

## 数据类型概览

MySQL 支持多种数据类型，主要分为以下几类：

- **数值类型** - 整数、浮点数、定点数
- **字符串类型** - 字符、文本、二进制数据
- **日期时间类型** - 日期、时间、时间戳
- **JSON 类型** - JSON 文档存储

## 数值类型

### 整数类型

| 类型      | 存储空间 | 有符号范围               | 无符号范围     | 常用场景        |
| --------- | -------- | ------------------------ | -------------- | --------------- |
| TINYINT   | 1 字节   | -128 ~ 127               | 0 ~ 255        | 状态值、年龄    |
| SMALLINT  | 2 字节   | -32768 ~ 32767           | 0 ~ 65535      | 小范围计数      |
| MEDIUMINT | 3 字节   | -8388608 ~ 8388607       | 0 ~ 16777215   | 中等计数        |
| INT       | 4 字节   | -2147483648 ~ 2147483647 | 0 ~ 4294967295 | 常规 ID、计数   |
| BIGINT    | 8 字节   | -2^63 ~ 2^63-1           | 0 ~ 2^64-1     | 大数值、雪花 ID |

#### 使用示例

```sql
CREATE TABLE user_info (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,           -- 主键用 BIGINT
    age TINYINT UNSIGNED NOT NULL,                   -- 年龄用 TINYINT
    view_count INT UNSIGNED DEFAULT 0,               -- 浏览量用 INT
    status TINYINT DEFAULT 1                         -- 状态值用 TINYINT
);
```

#### 最佳实践

> [!IMPORTANT]
>
> - 主键推荐使用 **BIGINT**，避免 ID 用尽
> - 状态字段使用 **TINYINT**，节省空间
> - 优先使用 **UNSIGNED** 无符号类型（如果不需要负数）
> - 避免使用 **MEDIUMINT**，不常用且容易混淆

### 浮点数和定点数

| 类型         | 存储空间 | 精度             | 用途       |
| ------------ | -------- | ---------------- | ---------- |
| FLOAT        | 4 字节   | 单精度，约 7 位  | 科学计算   |
| DOUBLE       | 8 字节   | 双精度，约 15 位 | 科学计算   |
| DECIMAL(M,D) | 变长     | 精确定点数       | 金额、价格 |

#### 使用示例

```sql
CREATE TABLE products (
    id INT PRIMARY KEY AUTO_INCREMENT,
    price DECIMAL(10, 2) NOT NULL,        -- 价格，最多10位，小数2位
    weight FLOAT,                          -- 重量（可以有误差）
    discount_rate DECIMAL(5, 4)           -- 折扣率 0.0000 ~ 9.9999
);

INSERT INTO products (price, weight, discount_rate)
VALUES (99.99, 1.5, 0.8500);
```

#### DECIMAL vs FLOAT

```sql
-- DECIMAL：精确存储
CREATE TABLE test_decimal (val DECIMAL(10, 2));
INSERT INTO test_decimal VALUES (123.456);  -- 存储为 123.46（四舍五入）
SELECT * FROM test_decimal;  -- 123.46

-- FLOAT：可能有误差
CREATE TABLE test_float (val FLOAT);
INSERT INTO test_float VALUES (123.456);
SELECT * FROM test_float;  -- 可能是 123.456001...
```

> [!WARNING] > **金额计算必须使用 DECIMAL**，不要使用 FLOAT 或 DOUBLE，否则可能出现精度丢失问题！

## 字符串类型

### 字符类型

| 类型       | 最大长度   | 存储方式         | 适用场景                   |
| ---------- | ---------- | ---------------- | -------------------------- |
| CHAR(M)    | 255 字符   | 定长，不足补空格 | 固定长度字符串（如手机号） |
| VARCHAR(M) | 65535 字节 | 变长，按实际长度 | 变长字符串（如用户名）     |

#### CHAR vs VARCHAR

```sql
CREATE TABLE string_test (
    fixed_str CHAR(10),      -- 定长
    var_str VARCHAR(10)      -- 变长
);

INSERT INTO string_test VALUES ('ABC', 'ABC');

-- CHAR(10) 存储：'ABC       ' (10字节，补7个空格)
-- VARCHAR(10) 存储：'ABC' (3字节 + 1字节长度)
```

#### 选择建议

- **CHAR** - 固定长度字符串，如：手机号、身份证号、MD5 值
- **VARCHAR** - 变长字符串，如：用户名、标题、描述

```sql
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,         -- 变长
    phone CHAR(11),                        -- 固定长度
    id_card CHAR(18),                      -- 固定长度
    email VARCHAR(100),                    -- 变长
    password CHAR(32)                      -- MD5固定32位
);
```

### 文本类型

| 类型       | 最大长度 | 适用场景         |
| ---------- | -------- | ---------------- |
| TINYTEXT   | 255 字节 | 很少使用         |
| TEXT       | 64 KB    | 文章摘要、短文本 |
| MEDIUMTEXT | 16 MB    | 文章内容         |
| LONGTEXT   | 4 GB     | 超长文本         |

#### 使用示例

```sql
CREATE TABLE articles (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(200) NOT NULL,
    summary TEXT,                    -- 摘要
    content MEDIUMTEXT NOT NULL,     -- 正文
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

> [!CAUTION]
> TEXT 类型字段：
>
> - 不能设置默认值
> - 不能作为主键
> - 索引需要指定长度
> - 会影响性能，尽量少用

### 二进制类型

| 类型         | 说明         |
| ------------ | ------------ |
| BINARY(M)    | 定长二进制   |
| VARBINARY(M) | 变长二进制   |
| BLOB         | 二进制大对象 |

```sql
CREATE TABLE files (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    file_hash BINARY(16),           -- MD5/UUID
    file_data BLOB                  -- 文件内容（不推荐存数据库）
);
```

> [!TIP]
> 不建议在数据库中存储文件内容，应该将文件存储在对象存储（如 OSS）中，数据库只存储文件路径。

## 日期时间类型

| 类型      | 格式                | 范围                    | 存储空间 | 时区 |
| --------- | ------------------- | ----------------------- | -------- | ---- |
| DATE      | YYYY-MM-DD          | 1000-01-01 ~ 9999-12-31 | 3 字节   | 无   |
| TIME      | HH:MM:SS            | -838:59:59 ~ 838:59:59  | 3 字节   | 无   |
| DATETIME  | YYYY-MM-DD HH:MM:SS | 1000-01-01 ~ 9999-12-31 | 8 字节   | 无关 |
| TIMESTAMP | YYYY-MM-DD HH:MM:SS | 1970-01-01 ~ 2038-01-19 | 4 字节   | 相关 |
| YEAR      | YYYY                | 1901 ~ 2155             | 1 字节   | 无   |

### 使用示例

```sql
CREATE TABLE events (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    event_date DATE,                                        -- 日期
    event_time TIME,                                        -- 时间
    event_datetime DATETIME,                                -- 日期时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,         -- 创建时间
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                         ON UPDATE CURRENT_TIMESTAMP        -- 更新时间
);

-- 插入数据
INSERT INTO events (event_date, event_time, event_datetime)
VALUES ('2025-12-09', '14:30:00', '2025-12-09 14:30:00');

-- 查询
SELECT * FROM events WHERE event_date >= '2025-01-01';
SELECT * FROM events WHERE created_at >= NOW() - INTERVAL 7 DAY;
```

### DATETIME vs TIMESTAMP

| 特性     | DATETIME    | TIMESTAMP      |
| -------- | ----------- | -------------- |
| 存储空间 | 8 字节      | 4 字节         |
| 时区     | 独立        | 依赖时区       |
| 范围     | 1000 ~ 9999 | 1970 ~ 2038    |
| 自动更新 | 不支持      | 支持 ON UPDATE |
| NULL     | 允许        | 不允许（默认） |

#### 使用建议

```sql
-- 推荐方式：记录时间使用 TIMESTAMP
CREATE TABLE orders (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    total_amount DECIMAL(10, 2),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    paid_at TIMESTAMP NULL                  -- 支付时间，可为空
);
```

> [!IMPORTANT]
>
> - 记录创建/更新时间用 **TIMESTAMP**
> - 业务日期（如生日）用 **DATE**
> - 统计时间段用 **DATETIME**

## JSON 类型

MySQL 5.7+ 支持原生 JSON 类型。

### 基本使用

```sql
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50),
    profile JSON,                -- JSON 字段
    settings JSON
);

-- 插入 JSON 数据
INSERT INTO users (username, profile, settings) VALUES
('张三',
 '{"age": 25, "city": "北京", "tags": ["Java", "MySQL"]}',
 '{"theme": "dark", "language": "zh-CN"}');

-- 查询 JSON 字段
SELECT
    username,
    profile->>'$.age' AS age,                    -- 25
    profile->>'$.city' AS city,                  -- 北京
    JSON_EXTRACT(profile, '$.tags') AS tags      -- ["Java", "MySQL"]
FROM users;

-- 查询 JSON 数组
SELECT username
FROM users
WHERE JSON_CONTAINS(profile, '"Java"', '$.tags');

-- 更新 JSON 字段
UPDATE users
SET profile = JSON_SET(profile, '$.age', 26)
WHERE username = '张三';
```

### JSON 函数

```sql
-- JSON_OBJECT：创建 JSON 对象
SELECT JSON_OBJECT('name', '张三', 'age', 25);
-- {"name": "张三", "age": 25}

-- JSON_ARRAY：创建 JSON 数组
SELECT JSON_ARRAY('Java', 'MySQL', 'Redis');
-- ["Java", "MySQL", "Redis"]

-- JSON_EXTRACT：提取值
SELECT JSON_EXTRACT('{"name": "张三"}', '$.name');
-- "张三"

-- JSON_SET：设置值
SELECT JSON_SET('{"age": 25}', '$.age', 26);
-- {"age": 26}

-- JSON_CONTAINS：判断是否包含
SELECT JSON_CONTAINS('["Java", "MySQL"]', '"Java"');
-- 1 (true)
```

> [!TIP]
> JSON 类型的优势：
>
> - 自动验证 JSON 格式
> - 比 TEXT 存储更高效
> - 支持丰富的 JSON 函数
> - 可以创建虚拟列索引

## 数据类型选择建议

### 常见场景推荐

| 场景     | 推荐类型         | 示例                                             |
| -------- | ---------------- | ------------------------------------------------ |
| 主键 ID  | BIGINT UNSIGNED  | `id BIGINT PRIMARY KEY AUTO_INCREMENT`           |
| 用户名   | VARCHAR(50)      | `username VARCHAR(50) NOT NULL`                  |
| 密码哈希 | CHAR(64)         | `password CHAR(64)` (SHA256)                     |
| 手机号   | CHAR(11)         | `phone CHAR(11)`                                 |
| 邮箱     | VARCHAR(100)     | `email VARCHAR(100)`                             |
| 金额     | DECIMAL(10,2)    | `price DECIMAL(10,2) NOT NULL`                   |
| 年龄     | TINYINT UNSIGNED | `age TINYINT UNSIGNED`                           |
| 状态     | TINYINT          | `status TINYINT DEFAULT 1`                       |
| 创建时间 | TIMESTAMP        | `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP` |
| 生日     | DATE             | `birthday DATE`                                  |
| 文章内容 | MEDIUMTEXT       | `content MEDIUMTEXT`                             |
| 配置信息 | JSON             | `settings JSON`                                  |

### 优化原则

> [!IMPORTANT] > **数据类型选择原则**:
>
> 1. **够用就好** - 不要浪费存储空间
> 2. **定长优于变长** - CHAR 比 VARCHAR 性能好（如果长度固定）
> 3. **整数优于字符串** - 用 TINYINT 存储状态，不要用 VARCHAR
> 4. **避免 NULL** - 尽量设置 NOT NULL 和默认值
> 5. **金额用 DECIMAL** - 不要用 FLOAT/DOUBLE

### 示例：规范的表设计

```sql
CREATE TABLE users (
    -- 主键
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '用户ID',

    -- 字符串字段
    username VARCHAR(50) NOT NULL COMMENT '用户名',
    email VARCHAR(100) NOT NULL COMMENT '邮箱',
    phone CHAR(11) DEFAULT NULL COMMENT '手机号',
    password CHAR(64) NOT NULL COMMENT '密码(SHA256)',

    -- 数值字段
    age TINYINT UNSIGNED DEFAULT 0 COMMENT '年龄',
    gender TINYINT DEFAULT 0 COMMENT '性别: 0-未知 1-男 2-女',
    balance DECIMAL(10,2) DEFAULT 0.00 COMMENT '余额',

    -- 时间字段
    birthday DATE DEFAULT NULL COMMENT '生日',
    last_login_at TIMESTAMP NULL COMMENT '最后登录时间',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                  ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    -- JSON字段
    profile JSON DEFAULT NULL COMMENT '个人资料',

    -- 索引
    UNIQUE KEY uk_username (username),
    UNIQUE KEY uk_email (email),
    KEY idx_phone (phone),
    KEY idx_created_at (created_at)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';
```

## 总结

本文详细介绍了 MySQL 的各种数据类型：

- ✅ 整数类型：主键用 BIGINT，状态用 TINYINT
- ✅ 浮点数：金额用 DECIMAL，科学计算用 DOUBLE
- ✅ 字符串：变长用 VARCHAR，定长用 CHAR
- ✅ 时间：记录时间用 TIMESTAMP，业务日期用 DATE
- ✅ JSON：存储灵活的结构化数据

掌握数据类型选择后，可以继续学习 [SQL 语法](./sql-syntax) 和 [索引优化](./indexes)！
