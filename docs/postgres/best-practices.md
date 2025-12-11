---
sidebar_position: 10
title: 最佳实践
---

# PostgreSQL 最佳实践

> [!IMPORTANT] > **实践指南**：本文档总结了 PostgreSQL 数据库设计、开发和运维的最佳实践，帮助你构建高性能、可维护的数据库系统。

## 数据库设计最佳实践

### 命名规范

#### 数据库命名

```sql
-- ✅ 推荐：使用下划线分隔的小写字母
CREATE DATABASE user_management;
CREATE DATABASE ecommerce_platform;

-- ❌ 避免：大小写混用、特殊字符
CREATE DATABASE UserManagement;
CREATE DATABASE ecommerce-platform;
```

#### 表命名

```sql
-- ✅ 推荐：复数形式，描述性强
CREATE TABLE users;
CREATE TABLE product_categories;
CREATE TABLE order_items;

-- ❌ 避免：缩写、前缀
CREATE TABLE usr;
CREATE TABLE tbl_user;
```

#### 字段命名

```sql
-- ✅ 推荐：清晰、描述性强
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email_address VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- ❌ 避免：太短或太长
CREATE TABLE users (
    uid INT,  -- 太短
    user_email_address_for_login VARCHAR(100),  -- 太长
    crtd TIMESTAMP  -- 缩写不清晰
);
```

### 表设计原则

#### 1. 每张表必须有主键

```sql
-- ✅ 推荐：使用 SERIAL 或 IDENTITY
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL
);

-- ✅ 或使用 IDENTITY
CREATE TABLE products (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- ✅ 使用 UUID（分布式系统）
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_name VARCHAR(100) NOT NULL
);
```

#### 2. 合理使用 NOT NULL

```sql
-- ✅ 推荐：必填字段使用 NOT NULL
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    nickname VARCHAR(50),  -- 可选字段允许 NULL
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

#### 3. 添加适当的默认值

```sql
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    status SMALLINT NOT NULL DEFAULT 1,  -- 1-正常 0-禁用
    login_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- 使用触发器自动更新 updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
```

#### 4. 添加字段注释

```sql
CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    order_no VARCHAR(32) NOT NULL,
    user_id BIGINT NOT NULL,
    total_amount NUMERIC(10,2) NOT NULL,
    status SMALLINT NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- 添加注释
COMMENT ON TABLE orders IS '订单表';
COMMENT ON COLUMN orders.id IS '订单ID';
COMMENT ON COLUMN orders.order_no IS '订单号';
COMMENT ON COLUMN orders.status IS '订单状态: 1-待支付 2-已支付 3-已取消';
```

### 数据类型选择

#### 整数类型

```sql
-- ✅ 根据实际范围选择合适的类型
CREATE TABLE users (
    id BIGSERIAL,              -- 大数据量主键
    age SMALLINT CHECK (age >= 0 AND age <= 150),  -- 年龄
    score SMALLINT,            -- 分数
    balance NUMERIC(10,2)      -- 金额使用 NUMERIC
);

-- ❌ 避免：浪费空间
CREATE TABLE users (
    age BIGINT,        -- 年龄不需要这么大的范围
    balance DOUBLE PRECISION  -- 金额应该用 NUMERIC
);
```

#### 字符串类型

```sql
-- ✅ 推荐：根据长度选择类型
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50),      -- 用户名有长度限制
    email VARCHAR(100),        -- 邮箱
    gender CHAR(1),           -- 固定长度用 CHAR
    bio TEXT,                 -- 长文本用 TEXT
    country_code CHAR(2)      -- ISO 国家代码
);

-- ❌ 避免：过大的 VARCHAR 或使用 CHAR 存储变长数据
CREATE TABLE users (
    username VARCHAR(255),    -- 用户名不需要这么长
    email CHAR(100)          -- 邮箱长度不固定，不应用 CHAR
);
```

#### 时间类型

```sql
-- ✅ 推荐
CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    event_date DATE,                           -- 只需要日期
    event_time TIME,                           -- 只需要时间
    created_at TIMESTAMP DEFAULT NOW(),        -- 时间戳
    created_at_tz TIMESTAMPTZ DEFAULT NOW()    -- 带时区的时间戳（推荐）
);

-- ❌ 避免：都用字符串
CREATE TABLE events (
    event_date VARCHAR(20),   -- 应该用 DATE
    created_at VARCHAR(30)    -- 应该用 TIMESTAMP
);
```

## 索引设计最佳实践

### 索引创建原则

#### 1. 为 WHERE 条件列创建索引

```sql
-- 经常查询的列
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- 复合索引（注意顺序）
CREATE INDEX idx_users_status_created ON users(status, created_at);
```

#### 2. 为 JOIN 列创建索引

```sql
-- 外键列应该有索引
CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id)
);

CREATE INDEX idx_orders_user_id ON orders(user_id);
```

#### 3. 为 ORDER BY 和 GROUP BY 列创建索引

```sql
-- 经常排序的列
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);

-- 经常分组的列
CREATE INDEX idx_products_category_id ON products(category_id);
```

### 复合索引设计

```sql
-- ✅ 推荐：遵循最左前缀原则
CREATE INDEX idx_orders_user_status_created
ON orders(user_id, status, created_at);

-- 可以利用这个索引的查询
-- WHERE user_id = 1
-- WHERE user_id = 1 AND status = 1
-- WHERE user_id = 1 AND status = 1 AND created_at > '2025-01-01'

-- ❌ 不能充分利用这个索引
-- WHERE status = 1  -- 跳过了第一列
-- WHERE created_at > '2025-01-01'  -- 跳过了前两列
```

### PostgreSQL 特色索引

#### 部分索引

```sql
-- 只为活跃用户创建索引
CREATE INDEX idx_active_users_username
ON users(username)
WHERE is_active = true;

-- 只为高价值订单创建索引
CREATE INDEX idx_high_value_orders
ON orders(created_at)
WHERE total > 1000;
```

#### 表达式索引

```sql
-- 为函数结果创建索引
CREATE INDEX idx_users_lower_email
ON users(LOWER(email));

-- 查询时可以使用这个索引
SELECT * FROM users WHERE LOWER(email) = 'user@example.com';

-- 为 JSON 字段创建索引
CREATE INDEX idx_users_metadata_name
ON users((metadata->>'name'));
```

#### GIN 索引（全文搜索、JSONB、数组）

```sql
-- JSONB 索引
CREATE INDEX idx_users_metadata ON users USING GIN(metadata);

-- 数组索引
CREATE INDEX idx_tags ON articles USING GIN(tags);

-- 全文搜索索引
CREATE INDEX idx_articles_search
ON articles USING GIN(to_tsvector('english', title || ' ' || content));
```

#### GiST 索引（范围类型、地理数据）

```sql
-- 范围类型索引
CREATE INDEX idx_bookings_period
ON bookings USING GIST(booking_period);

-- 地理数据索引（需要 PostGIS）
CREATE INDEX idx_locations_geom
ON locations USING GIST(geom);
```

### 索引维护

```sql
-- 查看索引使用情况
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan;

-- 查找未使用的索引
SELECT
    schemaname,
    tablename,
    indexname
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND indexname NOT LIKE '%pkey%';

-- 重建索引
REINDEX INDEX idx_users_email;
REINDEX TABLE users;
```

## 查询优化最佳实践

### SELECT 语句优化

```sql
-- ✅ 推荐：只查询需要的字段
SELECT id, username, email FROM users WHERE id = 1;

-- ❌ 避免：SELECT *
SELECT * FROM users WHERE id = 1;

-- ✅ 使用 LIMIT 限制结果集
SELECT * FROM users ORDER BY created_at DESC LIMIT 100;

-- ✅ 使用 EXISTS 代替 IN（大数据集）
SELECT * FROM users u
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);

-- ❌ 大表避免使用 IN
SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);
```

### WHERE 条件优化

```sql
-- ✅ 推荐：使用索引列的原始值
SELECT * FROM orders
WHERE created_at >= '2025-01-01 00:00:00'
  AND created_at < '2025-02-01 00:00:00';

-- ❌ 避免：在索引列上使用函数
SELECT * FROM orders WHERE DATE(created_at) = '2025-01-01';
SELECT * FROM orders WHERE EXTRACT(YEAR FROM created_at) = 2025;

-- ✅ 避免类型转换
SELECT * FROM users WHERE id = 123;  -- id 是整数

-- ❌ 隐式类型转换（会影响性能）
SELECT * FROM users WHERE id = '123';
```

### JOIN 优化

```sql
-- ✅ 推荐：明确 JOIN 类型
SELECT o.order_no, u.username
FROM orders o
INNER JOIN users u ON o.user_id = u.id
WHERE u.is_active = true;

-- ✅ 使用 JOIN 代替子查询
SELECT u.username, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.username;

-- ❌ 避免：使用子查询
SELECT u.username,
    (SELECT COUNT(*) FROM orders WHERE user_id = u.id) as order_count
FROM users u;
```

### 分页优化

```sql
-- ❌ 深分页性能差
SELECT * FROM orders ORDER BY id LIMIT 100000 OFFSET 100000;

-- ✅ 使用 Keyset 分页（游标分页）
SELECT * FROM orders
WHERE id > 100000
ORDER BY id
LIMIT 20;

-- ✅ 使用延迟关联
SELECT o.*
FROM orders o
INNER JOIN (
    SELECT id FROM orders ORDER BY id LIMIT 20 OFFSET 100000
) AS t ON o.id = t.id;
```

## 事务使用最佳实践

### 事务范围最小化

```sql
-- ✅ 推荐：保持事务简短
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- ❌ 避免：事务中包含不必要的操作
BEGIN;
SELECT * FROM products;  -- 不需要在事务中
-- ... 其他耗时操作
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;
```

### 使用适当的隔离级别

```sql
-- PostgreSQL 推荐使用 READ COMMITTED（默认）
SHOW transaction_isolation;

-- 需要更强隔离时使用 REPEATABLE READ
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- 执行查询
COMMIT;

-- 串行化（最高隔离级别，慎用）
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- 执行查询
COMMIT;
```

### 避免死锁

```sql
-- ✅ 推荐：按相同顺序访问资源
-- 事务1
UPDATE users SET status = 1 WHERE id = 1;
UPDATE orders SET status = 1 WHERE user_id = 1;

-- 事务2（相同顺序）
UPDATE users SET status = 1 WHERE id = 2;
UPDATE orders SET status = 1 WHERE user_id = 2;

-- ❌ 避免：不同顺序容易死锁
-- 事务1
UPDATE users SET status = 1 WHERE id = 1;
UPDATE orders SET status = 1 WHERE user_id = 1;

-- 事务2（相反顺序）
UPDATE orders SET status = 1 WHERE user_id = 1;
UPDATE users SET status = 1 WHERE id = 1;
```

## 安全性最佳实践

### 防止 SQL 注入

```java
// ✅ 推荐：使用预编译语句
PreparedStatement stmt = conn.prepareStatement(
    "SELECT * FROM users WHERE username = ? AND password = ?");
stmt.setString(1, username);
stmt.setString(2, password);

// ❌ 避免：字符串拼接
String sql = "SELECT * FROM users WHERE username = '" + username + "'";
```

### 权限管理

```sql
-- ✅ 推荐：最小权限原则
-- 创建只读用户
CREATE ROLE readonly;
GRANT CONNECT ON DATABASE mydb TO readonly;
GRANT USAGE ON SCHEMA public TO readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;

-- 应用程序用户
CREATE ROLE app_user WITH LOGIN PASSWORD 'password';
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;

-- ❌ 避免：过度授权
GRANT ALL PRIVILEGES ON DATABASE mydb TO app_user;  -- 太危险
```

### 行级安全（RLS）

```sql
-- 启用行级安全
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- 创建策略：用户只能看到自己的文档
CREATE POLICY user_documents ON documents
FOR SELECT
TO app_user
USING (user_id = current_user_id());

-- 创建策略：管理员可以看到所有文档
CREATE POLICY admin_documents ON documents
FOR ALL
TO admin_role
USING (true);
```

## 性能优化最佳实践

### 配置优化

```ini
# postgresql.conf 推荐配置

# 内存配置
shared_buffers = 4GB              # 系统内存的 25%
effective_cache_size = 12GB       # 系统内存的 75%
work_mem = 16MB                   # 单个操作的工作内存
maintenance_work_mem = 1GB        # 维护操作内存

# 连接配置
max_connections = 200             # 根据实际需求调整

# WAL 配置
wal_buffers = 16MB
checkpoint_timeout = 15min
max_wal_size = 2GB
min_wal_size = 1GB

# 查询优化
random_page_cost = 1.1            # SSD 建议设置为 1.1
effective_io_concurrency = 200    # SSD 可以设置更高

# 日志配置
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d.log'
log_min_duration_statement = 1000  # 记录超过1秒的查询
```

### 使用连接池

```bash
# 使用 PgBouncer
[databases]
mydb = host=localhost port=5432 dbname=mydb

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
```

### 定期维护

```sql
-- 手动 VACUUM
VACUUM ANALYZE users;

-- 查看表膨胀
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    n_dead_tup,
    n_live_tup,
    ROUND(n_dead_tup * 100.0 / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_ratio
FROM pg_stat_user_tables
WHERE n_dead_tup > 0
ORDER BY n_dead_tup DESC;

-- 更新统计信息
ANALYZE;
```

## 备份策略最佳实践

### 备份策略

```bash
# pg_dump 备份单个数据库
pg_dump -U postgres -d mydb -F c -b -v -f backup_$(date +%Y%m%d).dump

# pg_dumpall 备份所有数据库
pg_dumpall -U postgres > full_backup_$(date +%Y%m%d).sql

# 自动化备份脚本
#!/bin/bash
BACKUP_DIR="/backup/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -U postgres -d mydb -F c > ${BACKUP_DIR}/mydb_${DATE}.dump

# 删除30天前的备份
find ${BACKUP_DIR} -name "*.dump" -mtime +30 -delete
```

### WAL 归档（增量备份）

```sql
-- 配置 WAL 归档
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'
wal_level = replica
```

## 监控最佳实践

### 关键指标监控

```sql
-- 活跃连接数
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

-- 数据库大小
SELECT
    pg_database.datname,
    pg_size_pretty(pg_database_size(pg_database.datname)) AS size
FROM pg_database
ORDER BY pg_database_size(pg_database.datname) DESC;

-- 表大小
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;

-- 缓存命中率
SELECT
    sum(heap_blks_read) as heap_read,
    sum(heap_blks_hit) as heap_hit,
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as ratio
FROM pg_statio_user_tables;

-- 慢查询
SELECT
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

## 开发规范总结

> [!IMPORTANT] > **核心规范清单**：
>
> **表设计**
>
> - ✅ 每张表必须有主键
> - ✅ 使用 TIMESTAMPTZ 存储时间
> - ✅ 使用 TEXT 而非大 VARCHAR
> - ✅ 字段使用 NOT NULL + DEFAULT
> - ✅ 添加字段和表注释
>
> **索引设计**
>
> - ✅ 为 WHERE、JOIN、ORDER BY 列创建索引
> - ✅ 使用部分索引和表达式索引
> - ✅ 为 JSONB 和数组创建 GIN 索引
> - ✅ 定期检查并删除未使用的索引
>
> **SQL 编写**
>
> - ✅ 避免 SELECT \*
> - ✅ 使用 LIMIT 限制结果
> - ✅ WHERE 条件不使用函数
> - ✅ 使用预编译语句防止注入
>
> **性能优化**
>
> - ✅ 使用 EXPLAIN ANALYZE 分析查询
> - ✅ 启用 pg_stat_statements
> - ✅ 定期 VACUUM ANALYZE
> - ✅ 监控关键性能指标

## 总结

PostgreSQL 最佳实践涵盖：

- ✅ 数据库设计：命名规范、表设计、数据类型选择
- ✅ 索引设计：B-tree、GIN、GiST、部分索引、表达式索引
- ✅ 查询优化：SELECT、WHERE、JOIN、分页优化
- ✅ 事务使用：隔离级别、避免死锁
- ✅ 安全性：SQL 注入防护、权限管理、行级安全
- ✅ 性能优化：配置优化、连接池、定期维护
- ✅ 备份监控：pg_dump、WAL 归档、关键指标

继续学习 [实战案例](/docs/postgres/practical-examples) 和 [性能优化](/docs/postgres/performance-optimization)！
