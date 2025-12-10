---
sidebar_position: 10
title: 最佳实践
---

# MySQL 最佳实践

> [!IMPORTANT] > **实践指南**: 本文档总结了 MySQL 数据库设计、开发和运维的最佳实践，帮助你构建高性能、可维护的数据库系统。

## 数据库设计最佳实践

### 命名规范

#### 数据库命名

```sql
-- ✅ 推荐：使用下划线分隔的小写字母
CREATE DATABASE user_management;
CREATE DATABASE order_system;

-- ❌ 避免：大小写混用、特殊字符
CREATE DATABASE UserManagement;
CREATE DATABASE order-system;
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
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50),
    email_address VARCHAR(100),
    created_at TIMESTAMP,
    is_active TINYINT(1)
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
-- ✅ 推荐：使用自增主键
CREATE TABLE users (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL
);

-- ✅ 也可以：使用业务主键（如果确实唯一）
CREATE TABLE countries (
    country_code CHAR(2) PRIMARY KEY,
    country_name VARCHAR(100)
);
```

#### 2. 合理使用 NOT NULL

```sql
-- ✅ 推荐：必填字段使用 NOT NULL
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    nickname VARCHAR(50),  -- 可选字段允许 NULL
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

#### 3. 添加适当的默认值

```sql
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    status TINYINT NOT NULL DEFAULT 1 COMMENT '1-正常 0-禁用',
    login_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

#### 4. 添加字段注释

```sql
CREATE TABLE orders (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '订单ID',
    order_no VARCHAR(32) NOT NULL COMMENT '订单号',
    user_id BIGINT NOT NULL COMMENT '用户ID',
    total_amount DECIMAL(10,2) NOT NULL COMMENT '订单金额',
    status TINYINT NOT NULL DEFAULT 1 COMMENT '订单状态: 1-待支付 2-已支付 3-已取消',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单表';
```

### 数据类型选择

#### 整数类型

```sql
-- ✅ 根据实际范围选择合适的类型
CREATE TABLE users (
    id BIGINT UNSIGNED,  -- 用户ID可能很大
    age TINYINT UNSIGNED,  -- 年龄 0-255
    score SMALLINT,  -- 分数 -32768 ~ 32767
    balance DECIMAL(10,2)  -- 金额使用 DECIMAL
);

-- ❌ 避免：浪费空间
CREATE TABLE users (
    age BIGINT,  -- 年龄不需要这么大的范围
    score INT,  -- SMALLINT 就够了
    balance DOUBLE  -- 金额应该用 DECIMAL
);
```

#### 字符串类型

```sql
-- ✅ 推荐：根据长度选择类型
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    username VARCHAR(50),  -- 用户名固定长度范围
    email VARCHAR(100),  -- 邮箱
    gender CHAR(1),  -- 固定长度用 CHAR
    bio TEXT,  -- 长文本用 TEXT
    status ENUM('active', 'inactive', 'banned')  -- 固定值用 ENUM
);

-- ❌ 避免：过大的 VARCHAR
CREATE TABLE users (
    username VARCHAR(255),  -- 用户名不需要这么长
    gender VARCHAR(10)  -- 性别用 CHAR(1) 或 TINYINT
);
```

#### 时间类型

```sql
-- ✅ 推荐
CREATE TABLE events (
    id BIGINT PRIMARY KEY,
    event_date DATE,  -- 只需要日期
    event_time TIME,  -- 只需要时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 自动记录创建时间
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP  -- 自动更新
);

-- ❌ 避免：都用字符串
CREATE TABLE events (
    event_date VARCHAR(20),  -- 应该用 DATE
    created_at VARCHAR(30)  -- 应该用 TIMESTAMP
);
```

## 索引设计最佳实践

### 索引创建原则

#### 1. 为 WHERE 条件列创建索引

```sql
-- 经常查询的列
CREATE INDEX idx_username ON users(username);
CREATE INDEX idx_email ON users(email);

-- 复合索引（注意顺序）
CREATE INDEX idx_user_status_created ON users(status, created_at);
```

#### 2. 为 JOIN 列创建索引

```sql
-- 外键列应该有索引
CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    INDEX idx_user_id (user_id)
);
```

#### 3. 为 ORDER BY 和 GROUP BY 列创建索引

```sql
-- 经常排序的列
CREATE INDEX idx_created_at ON orders(created_at);

-- 经常分组的列
CREATE INDEX idx_category_id ON products(category_id);
```

### 复合索引设计

```sql
-- ✅ 推荐：遵循最左前缀原则
CREATE INDEX idx_user_status_created ON users(user_id, status, created_at);

-- 可以利用这个索引的查询
WHERE user_id = 1
WHERE user_id = 1 AND status = 1
WHERE user_id = 1 AND status = 1 AND created_at > '2025-01-01'

-- ❌ 不能利用这个索引
WHERE status = 1  -- 跳过了第一列
WHERE created_at > '2025-01-01'  -- 跳过了前两列
```

### 覆盖索引

```sql
-- ✅ 创建覆盖索引避免回表
CREATE INDEX idx_username_email ON users(username, email);

-- 查询只需要这两个字段，不需要回表
SELECT username, email FROM users WHERE username = '张三';
```

### 索引维护

```sql
-- 定期检查未使用的索引
SELECT * FROM sys.schema_unused_indexes;

-- 删除冗余索引
-- 如果有 (a, b) 的索引，通常不需要单独的 (a) 索引
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
SELECT * FROM orders WHERE YEAR(created_at) = 2025;

-- ✅ 避免类型转换
SELECT * FROM users WHERE id = 123;  -- id 是整数

-- ❌ 隐式类型转换
SELECT * FROM users WHERE id = '123';  -- 字符串转数字，索引可能失效
```

### JOIN 优化

```sql
-- ✅ 推荐：小表驱动大表
SELECT o.* FROM orders o
INNER JOIN users u ON o.user_id = u.id
WHERE u.status = 1;

-- ✅ 使用 STRAIGHT_JOIN 强制连接顺序（必要时）
SELECT STRAIGHT_JOIN o.* FROM users u
INNER JOIN orders o ON u.id = o.user_id;
```

### 分页优化

```sql
-- ❌ 深分页性能差
SELECT * FROM orders ORDER BY id LIMIT 100000, 20;

-- ✅ 使用延迟关联
SELECT o.* FROM orders o
INNER JOIN (
    SELECT id FROM orders ORDER BY id LIMIT 100000, 20
) AS t ON o.id = t.id;

-- ✅ 使用游标分页
SELECT * FROM orders WHERE id > 100000 ORDER BY id LIMIT 20;
```

## 事务使用最佳实践

### 事务范围最小化

```sql
-- ✅ 推荐：保持事务简短
START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- ❌ 避免：事务中包含不必要的操作
START TRANSACTION;
SELECT * FROM products;  -- 不需要在事务中
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- ... 其他耗时操作
COMMIT;
```

### 使用适当的隔离级别

```sql
-- 默认使用 REPEATABLE READ
-- 如果不需要可重复读，可以使用 READ COMMITTED
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- 查看当前隔离级别
SELECT @@transaction_isolation;
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
-- 应用程序只需要 SELECT, INSERT, UPDATE, DELETE
GRANT SELECT, INSERT, UPDATE, DELETE ON mydb.* TO 'app_user'@'localhost';

-- ❌ 避免：过度授权
GRANT ALL PRIVILEGES ON *.* TO 'app_user'@'%';  -- 太危险

-- ✅ 限制远程访问
CREATE USER 'admin'@'localhost' IDENTIFIED BY 'password';  -- 只允许本地

-- ❌ 避免：允许任意主机
CREATE USER 'admin'@'%' IDENTIFIED BY 'password';
```

### 密码安全

```sql
-- ✅ 使用强密码
CREATE USER 'user'@'localhost' IDENTIFIED BY 'Str0ng_P@ssw0rd_2025';

-- ✅ 定期更改密码
ALTER USER 'user'@'localhost' IDENTIFIED BY 'New_P@ssw0rd_2025';

-- ✅ 密码过期策略
ALTER USER 'user'@'localhost' PASSWORD EXPIRE INTERVAL 90 DAY;
```

## 性能优化最佳实践

### 配置优化

```ini
# my.cnf 推荐配置

[mysqld]
# 字符集
character-set-server=utf8mb4
collation-server=utf8mb4_unicode_ci

# InnoDB 缓冲池（最重要）
innodb_buffer_pool_size=8G  # 设置为物理内存的 50-80%
innodb_buffer_pool_instances=8

# 连接配置
max_connections=1000
max_connect_errors=100000

# 慢查询
slow_query_log=1
long_query_time=2
log_queries_not_using_indexes=1

# 二进制日志
log_bin=mysql-bin
binlog_format=ROW
expire_logs_days=7
```

### 表分区

```sql
-- 按时间范围分区
CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    order_date DATE NOT NULL,
    amount DECIMAL(10,2)
) PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);
```

### 定期维护

```sql
-- 分析表
ANALYZE TABLE users;

-- 优化表
OPTIMIZE TABLE users;

-- 检查表
CHECK TABLE users;

-- 修复表
REPAIR TABLE users;
```

## 备份策略最佳实践

### 备份策略

```bash
# 完整备份（每周）
mysqldump -u root -p --all-databases --single-transaction > full_backup_$(date +%Y%m%d).sql

# 增量备份（每天）
# 使用 binlog 进行增量备份

# 自动化备份脚本
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/mysql"
mysqldump -u root -p${MYSQL_PASSWORD} \
    --single-transaction \
    --routines \
    --triggers \
    --all-databases | gzip > ${BACKUP_DIR}/backup_${DATE}.sql.gz

# 删除30天前的备份
find ${BACKUP_DIR} -name "backup_*.sql.gz" -mtime +30 -delete
```

### 备份验证

```bash
# 定期测试恢复
mysql -u root -p test_db < backup.sql

# 验证备份文件完整性
gunzip -t backup.sql.gz
```

## 监控最佳实践

### 关键指标监控

```sql
-- QPS (每秒查询数)
SHOW GLOBAL STATUS LIKE 'Questions';

-- TPS (每秒事务数)
SHOW GLOBAL STATUS LIKE 'Com_commit';
SHOW GLOBAL STATUS LIKE 'Com_rollback';

-- 连接数
SHOW STATUS LIKE 'Threads_connected';

-- 慢查询
SHOW GLOBAL STATUS LIKE 'Slow_queries';

-- 表锁等待
SHOW GLOBAL STATUS LIKE 'Table_locks_waited';

-- InnoDB 缓冲池命中率
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read%';
```

### 报警设置

监控以下指标并设置报警：

- 连接数接近 max_connections
- 慢查询突然增多
- 主从复制延迟
- 磁盘空间不足
- CPU/内存使用率过高

## 开发规范总结

> [!IMPORTANT] > **核心规范清单**:
>
> **表设计**
>
> - ✅ 每张表必须有主键
> - ✅ 使用 InnoDB 存储引擎
> - ✅ 使用 utf8mb4 字符集
> - ✅ 字段使用 NOT NULL + DEFAULT
> - ✅ 添加字段和表注释
>
> **索引设计**
>
> - ✅ 为 WHERE、JOIN、ORDER BY 列创建索引
> - ✅ 单表索引数量不超过 5 个
> - ✅ 复合索引列数不超过 5 个
> - ✅ 遵循最左前缀原则
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
> - ✅ 使用 EXPLAIN 分析查询
> - ✅ 开启慢查询日志
> - ✅ 定期优化表
> - ✅ 监控关键性能指标

## 总结

本文档涵盖了 MySQL 开发和运维的最佳实践：

- ✅ 数据库设计：命名规范、表设计、数据类型选择
- ✅ 索引设计：创建原则、复合索引、覆盖索引
- ✅ 查询优化：SELECT、WHERE、JOIN、分页优化
- ✅ 事务使用：范围最小化、隔离级别、避免死锁
- ✅ 安全性：SQL 注入防护、权限管理、密码安全
- ✅ 性能优化：配置优化、分区、定期维护
- ✅ 备份监控：备份策略、关键指标监控

继续学习 [常见问题](/docs/mysql/faq) 和 [实战案例](/docs/mysql/practical-examples)！
