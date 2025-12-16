---
sidebar_position: 15
title: 分区表
---

# MySQL 分区表详解

> [!TIP] > **大表优化利器**: 分区表可以将大表按一定规则分割成多个小表，提升查询性能和管理效率。

## 分区表概述

### 什么是分区表

分区表将一张逻辑上的大表按照一定规则分割成多个物理分区，每个分区独立存储数据。

### 分区优势

- ✅ 提升查询性能（分区裁剪）
- ✅ 便于数据管理（按分区删除/归档）
- ✅ 可以跨磁盘存储
- ✅ 优化维护操作

### 分区限制

- ❌ 分区键必须是主键/唯一键的一部分
- ❌ 最多 8192 个分区
- ❌ 不支持外键
- ❌ 不支持全文索引

## 分区类型

### RANGE 分区

按连续范围分区，最常用的分区方式。

```sql
-- 按日期范围分区
CREATE TABLE orders (
    id BIGINT NOT NULL,
    customer_id INT NOT NULL,
    order_date DATE NOT NULL,
    amount DECIMAL(10,2),
    PRIMARY KEY (id, order_date)
) PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);

-- 按 ID 范围分区
CREATE TABLE users (
    id BIGINT NOT NULL PRIMARY KEY,
    username VARCHAR(50),
    created_at DATETIME
) PARTITION BY RANGE (id) (
    PARTITION p0 VALUES LESS THAN (1000000),
    PARTITION p1 VALUES LESS THAN (2000000),
    PARTITION p2 VALUES LESS THAN (3000000),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);
```

### RANGE COLUMNS 分区

支持多列和非整数类型。

```sql
-- 按日期列分区（无需函数转换）
CREATE TABLE logs (
    id BIGINT NOT NULL,
    log_date DATE NOT NULL,
    message TEXT,
    PRIMARY KEY (id, log_date)
) PARTITION BY RANGE COLUMNS (log_date) (
    PARTITION p202401 VALUES LESS THAN ('2024-02-01'),
    PARTITION p202402 VALUES LESS THAN ('2024-03-01'),
    PARTITION p202403 VALUES LESS THAN ('2024-04-01'),
    PARTITION pmax VALUES LESS THAN (MAXVALUE)
);

-- 多列分区
CREATE TABLE sales (
    id BIGINT NOT NULL,
    region VARCHAR(20) NOT NULL,
    sale_date DATE NOT NULL,
    amount DECIMAL(10,2),
    PRIMARY KEY (id, region, sale_date)
) PARTITION BY RANGE COLUMNS (region, sale_date) (
    PARTITION p_east_2024 VALUES LESS THAN ('EAST', '2025-01-01'),
    PARTITION p_west_2024 VALUES LESS THAN ('WEST', '2025-01-01'),
    PARTITION pmax VALUES LESS THAN (MAXVALUE, MAXVALUE)
);
```

### LIST 分区

按离散值列表分区。

```sql
-- 按地区分区
CREATE TABLE customers (
    id BIGINT NOT NULL,
    name VARCHAR(100),
    region_id INT NOT NULL,
    PRIMARY KEY (id, region_id)
) PARTITION BY LIST (region_id) (
    PARTITION p_north VALUES IN (1, 2, 3),
    PARTITION p_south VALUES IN (4, 5, 6),
    PARTITION p_east VALUES IN (7, 8, 9),
    PARTITION p_west VALUES IN (10, 11, 12)
);

-- LIST COLUMNS（支持字符串）
CREATE TABLE products (
    id BIGINT NOT NULL,
    name VARCHAR(100),
    category VARCHAR(20) NOT NULL,
    PRIMARY KEY (id, category)
) PARTITION BY LIST COLUMNS (category) (
    PARTITION p_electronics VALUES IN ('phone', 'laptop', 'tablet'),
    PARTITION p_clothing VALUES IN ('shirt', 'pants', 'shoes'),
    PARTITION p_food VALUES IN ('fruit', 'vegetable', 'meat')
);
```

### HASH 分区

按哈希值均匀分布数据。

```sql
-- 按用户 ID 哈希分区
CREATE TABLE user_actions (
    id BIGINT NOT NULL,
    user_id INT NOT NULL,
    action VARCHAR(50),
    created_at DATETIME,
    PRIMARY KEY (id, user_id)
) PARTITION BY HASH (user_id)
PARTITIONS 8;

-- LINEAR HASH（线性哈希，添加分区更快）
CREATE TABLE events (
    id BIGINT NOT NULL,
    event_type INT NOT NULL,
    data JSON,
    PRIMARY KEY (id, event_type)
) PARTITION BY LINEAR HASH (event_type)
PARTITIONS 4;
```

### KEY 分区

类似 HASH，但使用 MySQL 内部哈希函数。

```sql
-- KEY 分区（推荐用于字符串）
CREATE TABLE sessions (
    session_id VARCHAR(64) NOT NULL,
    user_id INT,
    data TEXT,
    expires_at DATETIME,
    PRIMARY KEY (session_id)
) PARTITION BY KEY (session_id)
PARTITIONS 16;
```

## 子分区

### 复合分区

```sql
-- RANGE + HASH 子分区
CREATE TABLE sales_log (
    id BIGINT NOT NULL,
    sale_date DATE NOT NULL,
    store_id INT NOT NULL,
    amount DECIMAL(10,2),
    PRIMARY KEY (id, sale_date, store_id)
) PARTITION BY RANGE (YEAR(sale_date))
SUBPARTITION BY HASH (store_id)
SUBPARTITIONS 4 (
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026)
);
```

## 分区管理

### 查看分区信息

```sql
-- 查看表的分区信息
SELECT
    PARTITION_NAME,
    PARTITION_ORDINAL_POSITION,
    PARTITION_METHOD,
    PARTITION_EXPRESSION,
    PARTITION_DESCRIPTION,
    TABLE_ROWS,
    AVG_ROW_LENGTH,
    DATA_LENGTH
FROM information_schema.PARTITIONS
WHERE TABLE_SCHEMA = 'your_database'
  AND TABLE_NAME = 'your_table';

-- 查看分区状态
SHOW CREATE TABLE orders\G
```

### 添加分区

```sql
-- RANGE 分区添加
ALTER TABLE orders ADD PARTITION (
    PARTITION p2026 VALUES LESS THAN (2027)
);

-- 添加多个分区
ALTER TABLE orders ADD PARTITION (
    PARTITION p2026 VALUES LESS THAN (2027),
    PARTITION p2027 VALUES LESS THAN (2028)
);

-- LIST 分区添加
ALTER TABLE customers ADD PARTITION (
    PARTITION p_central VALUES IN (13, 14, 15)
);

-- HASH/KEY 增加分区数
ALTER TABLE user_actions ADD PARTITION PARTITIONS 4;
-- 分区数从 8 增加到 12
```

### 删除分区

```sql
-- 删除分区（数据一起删除！）
ALTER TABLE orders DROP PARTITION p2022;

-- 删除多个分区
ALTER TABLE orders DROP PARTITION p2022, p2023;

-- HASH/KEY 减少分区数
ALTER TABLE user_actions COALESCE PARTITION 2;
-- 减少 2 个分区
```

### 重组分区

```sql
-- 拆分分区
ALTER TABLE orders REORGANIZE PARTITION pmax INTO (
    PARTITION p2026 VALUES LESS THAN (2027),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);

-- 合并分区
ALTER TABLE orders REORGANIZE PARTITION p2022, p2023 INTO (
    PARTITION p_old VALUES LESS THAN (2024)
);
```

### 交换分区

```sql
-- 创建同结构的普通表
CREATE TABLE orders_archive LIKE orders;
ALTER TABLE orders_archive REMOVE PARTITIONING;

-- 交换分区数据
ALTER TABLE orders EXCHANGE PARTITION p2022 WITH TABLE orders_archive;
-- p2022 分区的数据移到 orders_archive
-- orders_archive 表的数据移到 p2022（如果有的话）
```

### 清空分区

```sql
-- 清空特定分区的数据（比 DELETE 快很多）
ALTER TABLE orders TRUNCATE PARTITION p2022;

-- 清空多个分区
ALTER TABLE orders TRUNCATE PARTITION p2022, p2023;
```

## 分区裁剪

### 分区裁剪原理

当查询条件包含分区键时，MySQL 只扫描相关分区。

```sql
-- 查看分区裁剪效果
EXPLAIN SELECT * FROM orders WHERE order_date = '2024-06-15';

-- partitions 列显示只扫描 p2024 分区

-- 没有分区裁剪（扫描所有分区）
EXPLAIN SELECT * FROM orders WHERE amount > 1000;
```

### 优化分区裁剪

```sql
-- ✅ 良好的分区裁剪
SELECT * FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31';

-- ❌ 无法分区裁剪
SELECT * FROM orders WHERE YEAR(order_date) = 2024;

-- ✅ 改写后可以裁剪
SELECT * FROM orders
WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01';
```

## 分区表 vs 分表

| 特性       | 分区表     | 分表         |
| ---------- | ---------- | ------------ |
| 实现方式   | MySQL 内置 | 应用层实现   |
| 透明性     | 对应用透明 | 需要修改应用 |
| 跨分区查询 | 自动支持   | 需要手动处理 |
| 分布式支持 | 不支持     | 支持         |
| 数据量限制 | 单库限制   | 可无限扩展   |

### 选择建议

- **分区表**：单表数据量 < 50 亿，不需要分布式
- **分表**：数据量极大，需要分布式存储

## 最佳实践

> [!IMPORTANT] > **分区表最佳实践**:
>
> 1. ✅ 分区键选择高频查询条件
> 2. ✅ 使用 RANGE 分区管理时序数据
> 3. ✅ 定期维护分区（添加新分区、删除旧分区）
> 4. ✅ 确保查询条件包含分区键
> 5. ✅ 监控各分区数据分布
> 6. ❌ 避免过多分区（< 100 为宜）
> 7. ❌ 避免跨分区事务
> 8. ❌ 避免在分区键上使用函数

## 实战案例

### 日志表分区

```sql
-- 按月分区的日志表
CREATE TABLE app_logs (
    id BIGINT NOT NULL AUTO_INCREMENT,
    log_time DATETIME NOT NULL,
    level ENUM('DEBUG', 'INFO', 'WARN', 'ERROR') NOT NULL,
    message TEXT,
    PRIMARY KEY (id, log_time)
) PARTITION BY RANGE (TO_DAYS(log_time)) (
    PARTITION p202401 VALUES LESS THAN (TO_DAYS('2024-02-01')),
    PARTITION p202402 VALUES LESS THAN (TO_DAYS('2024-03-01')),
    PARTITION p202403 VALUES LESS THAN (TO_DAYS('2024-04-01')),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);

-- 定期添加分区的存储过程
DELIMITER //
CREATE PROCEDURE add_log_partition()
BEGIN
    DECLARE next_month DATE;
    DECLARE partition_name VARCHAR(20);
    DECLARE partition_value INT;

    SET next_month = DATE_ADD(LAST_DAY(NOW()), INTERVAL 1 DAY);
    SET partition_name = CONCAT('p', DATE_FORMAT(next_month, '%Y%m'));
    SET partition_value = TO_DAYS(DATE_ADD(next_month, INTERVAL 1 MONTH));

    SET @sql = CONCAT('ALTER TABLE app_logs REORGANIZE PARTITION pmax INTO (',
        'PARTITION ', partition_name, ' VALUES LESS THAN (', partition_value, '),',
        'PARTITION pmax VALUES LESS THAN MAXVALUE)');

    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
END//
DELIMITER ;

-- 定期删除旧分区
ALTER TABLE app_logs DROP PARTITION p202401;
```

### 订单表分区

```sql
-- 按年份分区的订单表
CREATE TABLE orders (
    id BIGINT NOT NULL AUTO_INCREMENT,
    order_no VARCHAR(32) NOT NULL,
    user_id BIGINT NOT NULL,
    order_date DATE NOT NULL,
    status TINYINT NOT NULL DEFAULT 0,
    total_amount DECIMAL(12,2),
    PRIMARY KEY (id, order_date),
    UNIQUE KEY uk_order_no (order_no, order_date),
    KEY idx_user_id (user_id, order_date)
) PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);
```

## 总结

本文介绍了 MySQL 分区表：

- ✅ 分区类型：RANGE、LIST、HASH、KEY
- ✅ 子分区和复合分区
- ✅ 分区管理操作
- ✅ 分区裁剪优化
- ✅ 分区表 vs 分表
- ✅ 最佳实践和实战案例

继续学习 [性能优化](/docs/mysql/performance-optimization) 和 [实战案例](/docs/mysql/practical-examples)！
