---
sidebar_position: 7
title: 性能优化
---

# MySQL 性能优化

> [!TIP] > **性能调优核心**: 查询优化、索引设计、参数配置是提升 MySQL 性能的三大支柱。本文提供实用的优化技巧和最佳实践。

## 查询优化

### EXPLAIN 执行计划分析

使用 EXPLAIN 分析查询性能是优化的第一步。

```sql
EXPLAIN SELECT * FROM orders WHERE user_id = 1000;
```

### 慢查询日志

```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = ON;
SET GLOBAL long_query_time = 2;  -- 2秒以上为慢查询

-- 查看慢查询日志位置
SHOW VARIABLES LIKE 'slow_query_log_file';

-- 分析慢查询日志
-- 使用 mysqldumpslow 工具
mysqldumpslow -s t -t 10 /path/to/slow.log
```

### 查询优化技巧

#### 1. 避免 SELECT \*

```sql
-- ❌ 不推荐
SELECT * FROM users WHERE id = 1;

-- ✅ 推荐：只查询需要的字段
SELECT id, username, email FROM users WHERE id = 1;
```

#### 2. 使用 LIMIT 限制结果

```sql
-- ❌ 不推荐：全表扫描
SELECT * FROM users ORDER BY created_at DESC;

-- ✅ 推荐：限制结果数量
SELECT * FROM users ORDER BY created_at DESC LIMIT 100;
```

#### 3. 避免在 WHERE 中使用函数

```sql
-- ❌ 索引失效
SELECT * FROM users WHERE DATE(created_at) = '2025-12-09';

-- ✅ 改写为范围查询
SELECT * FROM users
WHERE created_at >= '2025-12-09 00:00:00'
  AND created_at < '2025-12-10 00:00:00';
```

#### 4. 使用 JOIN 代替子查询

```sql
-- ❌ 子查询性能较差
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE status = 'paid');

-- ✅ JOIN 性能更好
SELECT DISTINCT u.* FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.status = 'paid';
```

#### 5. 优化分页查询

```sql
-- ❌ 深分页性能差
SELECT * FROM users ORDER BY id LIMIT 100000, 10;

-- ✅ 使用上次最大 ID
SELECT * FROM users WHERE id > 100000 ORDER BY id LIMIT 10;

-- ✅ 使用子查询优化
SELECT * FROM users
WHERE id >= (SELECT id FROM users ORDER BY id LIMIT 100000, 1)
ORDER BY id LIMIT 10;
```

## 索引优化

### 索引设计原则

> [!IMPORTANT] > **索引优化要点**:
>
> 1. 为 WHERE、JOIN、ORDER BY 列创建索引
> 2. 选择性高的列优先
> 3. 使用复合索引（注意最左前缀原则）
> 4. 利用覆盖索引
> 5. 删除冗余和未使用的索引

### 索引监控

```sql
-- 查看索引使用情况
SELECT * FROM sys.schema_unused_indexes;

-- 查看索引大小
SELECT
    table_name,
    index_name,
    ROUND(stat_value * @@innodb_page_size / 1024 / 1024, 2) AS size_mb
FROM mysql.innodb_index_stats
WHERE stat_name = 'size'
ORDER BY size_mb DESC;
```

## 表结构优化

### 数据类型选择

```sql
-- ✅ 合适的数据类型
CREATE TABLE users (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    age TINYINT UNSIGNED,
    status TINYINT DEFAULT 1,
    balance DECIMAL(10,2)
);

-- ❌ 浪费空间
CREATE TABLE users_bad (
    id BIGINT,  -- 没有 UNSIGNED
    age INT,    -- TINYINT 就够了
    status VARCHAR(10),  -- 应该用 TINYINT
    balance DOUBLE  -- 金额应该用 DECIMAL
);
```

### 表分区

```sql
-- 范围分区
CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    order_date DATE,
    amount DECIMAL(10,2)
) PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026)
);

-- 哈希分区
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    username VARCHAR(50)
) PARTITION BY HASH(id) PARTITIONS 4;

-- 列表分区
CREATE TABLE sales (
    id BIGINT PRIMARY KEY,
    region VARCHAR(20)
) PARTITION BY LIST COLUMNS(region) (
    PARTITION p_north VALUES IN ('北京', '天津'),
    PARTITION p_south VALUES IN ('上海', '深圳')
);
```

## 参数配置优化

### 连接相关

```sql
-- 最大连接数
SET GLOBAL max_connections = 1000;

-- 连接超时
SET GLOBAL wait_timeout = 28800;  -- 8小时
SET GLOBAL interactive_timeout = 28800;
```

### 缓冲池配置

```sql
-- InnoDB 缓冲池大小（最重要的参数）
-- 建议设置为物理内存的 50-80%
SET GLOBAL innodb_buffer_pool_size = 8589934592;  -- 8GB

-- 缓冲池实例数
SET GLOBAL innodb_buffer_pool_instances = 8;
```

### 日志配置

```sql
-- Redo Log 大小
SET GLOBAL innodb_log_file_size = 1073741824;  -- 1GB

-- Binlog 配置
SET GLOBAL binlog_cache_size = 1048576;  -- 1MB
SET GLOBAL max_binlog_size = 1073741824;  -- 1GB
```

### 查询缓存（MySQL 5.7，8.0 已移除）

```sql
-- 查询缓存大小
SET GLOBAL query_cache_size = 268435456;  -- 256MB

-- 查询缓存类型
SET GLOBAL query_cache_type = 1;  -- 1=ON, 0=OFF, 2=DEMAND
```

## 硬件和系统优化

### 服务器配置建议

| 组件 | 推荐配置                    |
| ---- | --------------------------- |
| CPU  | 多核心，频率越高越好        |
| 内存 | 越大越好，建议 16GB+        |
| 磁盘 | SSD（固态硬盘）性能远超 HDD |
| 网络 | 千兆以上网卡                |

### 操作系统优化

```bash
# Linux 优化
# 1. 调整文件描述符限制
ulimit -n 65535

# 2. 关闭 SWAP
swapoff -a

# 3. 调整 TCP 参数
sysctl -w net.ipv4.tcp_max_syn_backlog=4096
sysctl -w net.core.somaxconn=4096
```

## 分库分表

### 垂直分表

将一张表的列拆分到多张表。

```sql
-- 原表
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    username VARCHAR(50),
    email VARCHAR(100),
    profile TEXT,  -- 大字段
    settings JSON  -- 大字段
);

-- 拆分后
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    username VARCHAR(50),
    email VARCHAR(100)
);

CREATE TABLE user_details (
    user_id BIGINT PRIMARY KEY,
    profile TEXT,
    settings JSON,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### 水平分表

将一张表的行拆分到多张表。

```sql
-- 按用户 ID 范围分表
CREATE TABLE users_1 (id BIGINT PRIMARY KEY, ...) -- id: 1-1000000
CREATE TABLE users_2 (id BIGINT PRIMARY KEY, ...) -- id: 1000001-2000000

-- 按时间分表
CREATE TABLE orders_202501 (order_date DATE, ...)
CREATE TABLE orders_202502 (order_date DATE, ...)
```

### 分库

将不同业务的表分到不同数据库。

```sql
-- 用户库
CREATE DATABASE user_db;
USE user_db;
CREATE TABLE users (...);

-- 订单库
CREATE DATABASE order_db;
USE order_db;
CREATE TABLE orders (...);

-- 产品库
CREATE DATABASE product_db;
USE product_db;
CREATE TABLE products (...);
```

## 读写分离

### 主从复制

```sql
-- 主库配置 (my.cnf)
[mysqld]
server-id=1
log-bin=mysql-bin
binlog-format=ROW

-- 从库配置 (my.cnf)
[mysqld]
server-id=2
relay-log=mysql-relay-bin

-- 从库配置主从复制
CHANGE MASTER TO
    MASTER_HOST='master_host',
    MASTER_USER='repl_user',
    MASTER_PASSWORD='password',
    MASTER_LOG_FILE='mysql-bin.000001',
    MASTER_LOG_POS=154;

START SLAVE;

-- 查看从库状态
SHOW SLAVE STATUS\G
```

### 应用层读写分离

```java
// Java 示例
// 写操作 -> 主库
userMapper.insert(user);

// 读操作 -> 从库
List<User> users = userMapper.selectAll();
```

## 监控和分析

### 性能监控

```sql
-- 查看当前连接
SHOW PROCESSLIST;

-- 查看 InnoDB 状态
SHOW ENGINE INNODB STATUS\G

-- 查看表状态
SHOW TABLE STATUS FROM database_name;

-- 查看索引统计
SHOW INDEX FROM table_name;
```

### 性能指标

```sql
-- QPS (Queries Per Second)
SHOW GLOBAL STATUS LIKE 'Questions';

-- TPS (Transactions Per Second)
SHOW GLOBAL STATUS LIKE 'Com_commit';
SHOW GLOBAL STATUS LIKE 'Com_rollback';

-- 缓冲池命中率
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read%';

-- 慢查询数量
SHOW GLOBAL STATUS LIKE 'Slow_queries';
```

## 性能测试工具

### sysbench

```bash
# 安装
apt-get install sysbench

# 准备测试数据
sysbench /usr/share/sysbench/oltp_read_write.lua \
    --mysql-host=localhost \
    --mysql-user=root \
    --mysql-password=password \
    --mysql-db=test \
    --tables=10 \
    --table-size=100000 \
    prepare

# 运行测试
sysbench /usr/share/sysbench/oltp_read_write.lua \
    --mysql-host=localhost \
    --mysql-user=root \
    --mysql-password=password \
    --mysql-db=test \
    --tables=10 \
    --table-size=100000 \
    --threads=16 \
    --time=60 \
    run

# 清理
sysbench ... cleanup
```

## 最佳实践总结

> [!IMPORTANT] > **性能优化清单**:
>
> **查询优化**
>
> - ✅ 使用 EXPLAIN 分析执行计划
> - ✅ 避免 SELECT \*
> - ✅ 使用 LIMIT 限制结果
> - ✅ WHERE 条件避免使用函数
> - ✅ 优化分页查询
>
> **索引优化**
>
> - ✅ 为 WHERE、JOIN 列创建索引
> - ✅ 使用复合索引（最左前缀）
> - ✅ 利用覆盖索引
> - ✅ 删除冗余索引
>
> **配置优化**
>
> - ✅ 合理设置 innodb_buffer_pool_size
> - ✅ 调整连接数和超时参数
> - ✅ 开启慢查询日志
>
> **架构优化**
>
> - ✅ 读写分离
> - ✅ 分库分表
> - ✅ 使用缓存（Redis）

## 总结

本文介绍了 MySQL 性能优化的核心内容：

- ✅ 查询优化：EXPLAIN、慢查询、优化技巧
- ✅ 索引优化：设计原则、监控维护
- ✅ 表结构优化：数据类型、分区
- ✅ 参数配置：缓冲池、日志、连接
- ✅ 架构优化：分库分表、读写分离
- ✅ 监控分析：性能指标、测试工具

继续学习 [存储过程](/docs/mysql/stored-procedures) 和 [备份恢复](/docs/mysql/backup-recovery)！
