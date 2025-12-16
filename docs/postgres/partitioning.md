---
sidebar_position: 14
title: 分区表
---

# PostgreSQL 分区表

分区表是将大表拆分为多个小表的技术，可以显著提升查询性能和管理效率。

## 📚 分区类型

PostgreSQL 支持三种分区方式：

| 类型      | 描述           | 适用场景         |
| --------- | -------------- | ---------------- |
| **RANGE** | 基于范围分区   | 时间序列数据     |
| **LIST**  | 基于值列表分区 | 按地区、类别分类 |
| **HASH**  | 基于哈希值分区 | 均匀分布数据     |

## 📅 范围分区（RANGE）

最常用的分区方式，适合时间序列数据。

### 1. 创建分区表

```sql
-- 创建父表
CREATE TABLE orders (
    id SERIAL,
    customer_id INT NOT NULL,
    order_date DATE NOT NULL,
    total NUMERIC(10, 2),
    status VARCHAR(20),
    PRIMARY KEY (id, order_date)
) PARTITION BY RANGE (order_date);

-- 创建子分区
CREATE TABLE orders_2024_q1 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE orders_2024_q2 PARTITION OF orders
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

CREATE TABLE orders_2024_q3 PARTITION OF orders
    FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');

CREATE TABLE orders_2024_q4 PARTITION OF orders
    FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');
```

### 2. 按月分区

```sql
CREATE TABLE logs (
    id BIGSERIAL,
    log_time TIMESTAMP NOT NULL,
    level VARCHAR(10),
    message TEXT,
    PRIMARY KEY (id, log_time)
) PARTITION BY RANGE (log_time);

-- 2024 年各月分区
CREATE TABLE logs_2024_01 PARTITION OF logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE logs_2024_02 PARTITION OF logs
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- ... 继续创建其他月份
```

### 3. 默认分区

```sql
-- 创建默认分区，存储不匹配任何分区的数据
CREATE TABLE orders_default PARTITION OF orders DEFAULT;
```

## 📋 列表分区（LIST）

适合按固定分类存储数据。

### 1. 按地区分区

```sql
CREATE TABLE customers (
    id SERIAL,
    name VARCHAR(100),
    email VARCHAR(100),
    region VARCHAR(20) NOT NULL,
    PRIMARY KEY (id, region)
) PARTITION BY LIST (region);

CREATE TABLE customers_east PARTITION OF customers
    FOR VALUES IN ('beijing', 'shanghai', 'guangzhou');

CREATE TABLE customers_west PARTITION OF customers
    FOR VALUES IN ('chengdu', 'chongqing', 'xian');

CREATE TABLE customers_other PARTITION OF customers DEFAULT;
```

### 2. 按状态分区

```sql
CREATE TABLE tasks (
    id SERIAL,
    title VARCHAR(200),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, status)
) PARTITION BY LIST (status);

CREATE TABLE tasks_pending PARTITION OF tasks
    FOR VALUES IN ('pending', 'new');

CREATE TABLE tasks_active PARTITION OF tasks
    FOR VALUES IN ('in_progress', 'review');

CREATE TABLE tasks_done PARTITION OF tasks
    FOR VALUES IN ('completed', 'archived');
```

## 🔢 哈希分区（HASH）

适合需要均匀分布数据的场景。

### 1. 创建哈希分区

```sql
CREATE TABLE events (
    id BIGSERIAL,
    user_id INT NOT NULL,
    event_type VARCHAR(50),
    event_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, user_id)
) PARTITION BY HASH (user_id);

-- 创建 4 个分区
CREATE TABLE events_0 PARTITION OF events
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);

CREATE TABLE events_1 PARTITION OF events
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);

CREATE TABLE events_2 PARTITION OF events
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);

CREATE TABLE events_3 PARTITION OF events
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

## 🏗️ 多级分区

可以组合多种分区方式。

```sql
-- 先按时间范围分区，再按地区列表分区
CREATE TABLE sales (
    id SERIAL,
    sale_date DATE NOT NULL,
    region VARCHAR(20) NOT NULL,
    amount NUMERIC(10, 2),
    PRIMARY KEY (id, sale_date, region)
) PARTITION BY RANGE (sale_date);

-- 创建 2024 年分区
CREATE TABLE sales_2024 PARTITION OF sales
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')
    PARTITION BY LIST (region);

-- 在 2024 年下创建地区分区
CREATE TABLE sales_2024_east PARTITION OF sales_2024
    FOR VALUES IN ('beijing', 'shanghai');

CREATE TABLE sales_2024_west PARTITION OF sales_2024
    FOR VALUES IN ('chengdu', 'chongqing');
```

## 🔧 分区管理

### 1. 查看分区信息

```sql
-- 查看分区表结构
SELECT
    parent.relname AS parent_table,
    child.relname AS partition_name,
    pg_get_expr(child.relpartbound, child.oid) AS partition_expression
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid
WHERE parent.relname = 'orders';

-- 查看分区大小
SELECT
    child.relname AS partition_name,
    pg_size_pretty(pg_relation_size(child.oid)) AS size
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid
WHERE parent.relname = 'orders'
ORDER BY child.relname;
```

### 2. 添加新分区

```sql
-- 添加新季度分区
CREATE TABLE orders_2025_q1 PARTITION OF orders
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
```

### 3. 删除分区

```sql
-- 方式 1：删除分区及数据
DROP TABLE orders_2023_q1;

-- 方式 2：分离分区（保留数据）
ALTER TABLE orders DETACH PARTITION orders_2023_q1;

-- 之后可以单独删除或归档
DROP TABLE orders_2023_q1;
-- 或
-- ALTER TABLE orders_2023_q1 RENAME TO orders_2023_q1_archive;
```

### 4. 合并分区

```sql
-- 分离要合并的分区
ALTER TABLE orders DETACH PARTITION orders_2024_q1;
ALTER TABLE orders DETACH PARTITION orders_2024_q2;

-- 创建新的合并分区
CREATE TABLE orders_2024_h1 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-07-01');

-- 迁移数据
INSERT INTO orders_2024_h1 SELECT * FROM orders_2024_q1;
INSERT INTO orders_2024_h1 SELECT * FROM orders_2024_q2;

-- 删除旧分区
DROP TABLE orders_2024_q1;
DROP TABLE orders_2024_q2;
```

## 📊 索引和约束

### 1. 分区表索引

```sql
-- 在父表上创建索引，自动应用到所有分区
CREATE INDEX idx_orders_customer_id ON orders (customer_id);
CREATE INDEX idx_orders_status ON orders (status);

-- 在特定分区上创建索引
CREATE INDEX idx_orders_2024_q1_customer
    ON orders_2024_q1 (customer_id);
```

### 2. 唯一约束

```sql
-- 主键必须包含分区键
CREATE TABLE orders (
    id SERIAL,
    order_date DATE NOT NULL,
    -- id 和 order_date 都是主键的一部分
    PRIMARY KEY (id, order_date)
) PARTITION BY RANGE (order_date);

-- 唯一约束也必须包含分区键
ALTER TABLE orders ADD CONSTRAINT uk_orders_code
    UNIQUE (order_code, order_date);
```

## ⚡ 分区裁剪

PostgreSQL 会自动进行分区裁剪（Partition Pruning），只扫描相关分区。

```sql
-- 确保启用分区裁剪
SET enable_partition_pruning = on;

-- 查询时只扫描 2024_q1 分区
EXPLAIN SELECT * FROM orders
WHERE order_date = '2024-02-15';

-- 输出示例：
-- Seq Scan on orders_2024_q1 orders
--   Filter: (order_date = '2024-02-15'::date)
```

## 🔄 自动分区管理

### 使用 pg_partman 扩展

```sql
-- 安装扩展
CREATE EXTENSION pg_partman;

-- 创建分区父表
CREATE TABLE events (
    id BIGSERIAL,
    event_time TIMESTAMPTZ NOT NULL,
    data JSONB
) PARTITION BY RANGE (event_time);

-- 配置自动分区
SELECT partman.create_parent(
    p_parent_table => 'public.events',
    p_control => 'event_time',
    p_type => 'native',
    p_interval => 'daily',
    p_premake => 7  -- 提前创建 7 天的分区
);

-- 运行维护（通常通过 cron 定时执行）
SELECT partman.run_maintenance();
```

## 💡 最佳实践

1. **选择合适的分区键**

   - 查询中经常使用的列
   - 数据分布均匀的列
   - 不频繁更新的列

2. **分区大小适中**

   - 建议每个分区 1GB - 100GB
   - 分区数量不宜过多（< 1000）

3. **主键包含分区键**

   - PostgreSQL 的限制
   - 保证数据唯一性

4. **使用默认分区**

   - 避免数据插入失败
   - 定期检查默认分区

5. **定期维护**
   - 删除过期分区
   - 提前创建新分区

## 📚 相关资源

- [索引优化](/docs/postgres/indexes) - 分区表索引
- [性能优化](/docs/postgres/performance-optimization) - 查询优化
- [备份恢复](/docs/postgres/backup-recovery) - 分区表备份
