---
sidebar_position: 9
title: 快速参考
---

# MySQL 快速参考手册

> [!TIP] > **快速查找**: 本文档提供 MySQL 常用命令、数据类型、函数和配置参数的快速参考，方便日常开发查询使用。

## 常用命令速查

### 数据库操作

```sql
-- 创建数据库
CREATE DATABASE dbname CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 删除数据库
DROP DATABASE dbname;

-- 查看所有数据库
SHOW DATABASES;

-- 使用数据库
USE dbname;

-- 查看当前数据库
SELECT DATABASE();
```

### 表操作

```sql
-- 创建表
CREATE TABLE table_name (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 查看所有表
SHOW TABLES;

-- 查看表结构
DESC table_name;
SHOW COLUMNS FROM table_name;

-- 查看建表语句
SHOW CREATE TABLE table_name;

-- 删除表
DROP TABLE table_name;

-- 清空表
TRUNCATE TABLE table_name;

-- 重命名表
RENAME TABLE old_name TO new_name;
```

### 索引操作

```sql
-- 创建索引
CREATE INDEX idx_name ON table_name(column_name);
CREATE UNIQUE INDEX idx_name ON table_name(column_name);

-- 创建复合索引
CREATE INDEX idx_name ON table_name(col1, col2);

-- 删除索引
DROP INDEX idx_name ON table_name;

-- 查看索引
SHOW INDEX FROM table_name;
```

### CRUD 操作

```sql
-- 插入数据
INSERT INTO users (name, email) VALUES ('张三', 'zhangsan@example.com');

-- 批量插入
INSERT INTO users (name, email) VALUES
    ('张三', 'zhangsan@example.com'),
    ('李四', 'lisi@example.com');

-- 查询数据
SELECT * FROM users WHERE id = 1;
SELECT name, email FROM users WHERE age > 18;

-- 更新数据
UPDATE users SET age = 25 WHERE id = 1;

-- 删除数据
DELETE FROM users WHERE id = 1;
```

### 用户和权限

```sql
-- 创建用户
CREATE USER 'username'@'localhost' IDENTIFIED BY 'password';

-- 授权
GRANT ALL PRIVILEGES ON dbname.* TO 'username'@'localhost';
GRANT SELECT, INSERT ON dbname.table_name TO 'username'@'localhost';

-- 撤销权限
REVOKE ALL PRIVILEGES ON dbname.* FROM 'username'@'localhost';

-- 刷新权限
FLUSH PRIVILEGES;

-- 查看用户权限
SHOW GRANTS FOR 'username'@'localhost';

-- 删除用户
DROP USER 'username'@'localhost';
```

## 数据类型速查表

### 数值类型

| 类型          | 字节 | 范围                           | 说明         |
| ------------- | ---- | ------------------------------ | ------------ |
| TINYINT       | 1    | -128 ~ 127 (UNSIGNED: 0 ~ 255) | 极小整数     |
| SMALLINT      | 2    | -32768 ~ 32767                 | 小整数       |
| MEDIUMINT     | 3    | -8388608 ~ 8388607             | 中整数       |
| INT           | 4    | -2147483648 ~ 2147483647       | 整数         |
| BIGINT        | 8    | -2^63 ~ 2^63-1                 | 大整数       |
| DECIMAL(M, D) | 变长 | 依赖 M 和 D                    | 精确小数     |
| FLOAT         | 4    | -3.4E+38 ~ 3.4E+38             | 单精度浮点数 |
| DOUBLE        | 8    | -1.8E+308 ~ 1.8E+308           | 双精度浮点数 |

### 字符串类型

| 类型       | 最大长度   | 说明         |
| ---------- | ---------- | ------------ |
| CHAR(M)    | 255        | 定长字符串   |
| VARCHAR(M) | 65535      | 变长字符串   |
| TINYTEXT   | 255        | 短文本       |
| TEXT       | 65535      | 文本         |
| MEDIUMTEXT | 16777215   | 中等长度文本 |
| LONGTEXT   | 4294967295 | 长文本       |
| ENUM       | 65535      | 枚举         |
| SET        | 64         | 集合         |

### 日期时间类型

| 类型      | 格式                | 范围                                      |
| --------- | ------------------- | ----------------------------------------- |
| DATE      | YYYY-MM-DD          | 1000-01-01 ~ 9999-12-31                   |
| TIME      | HH:MM:SS            | -838:59:59 ~ 838:59:59                    |
| DATETIME  | YYYY-MM-DD HH:MM:SS | 1000-01-01 00:00:00 ~ 9999-12-31 23:59:59 |
| TIMESTAMP | YYYY-MM-DD HH:MM:SS | 1970-01-01 00:00:01 ~ 2038-01-19 03:14:07 |
| YEAR      | YYYY                | 1901 ~ 2155                               |

## 常用函数速查

### 字符串函数

```sql
-- 拼接字符串
CONCAT('Hello', ' ', 'World')  -- 'Hello World'

-- 获取长度
LENGTH('Hello')  -- 5
CHAR_LENGTH('你好')  -- 2

-- 转换大小写
UPPER('hello')  -- 'HELLO'
LOWER('HELLO')  -- 'hello'

-- 截取字符串
SUBSTRING('Hello World', 1, 5)  -- 'Hello'
LEFT('Hello', 2)  -- 'He'
RIGHT('Hello', 2)  -- 'lo'

-- 去除空格
TRIM('  hello  ')  -- 'hello'
LTRIM('  hello')  -- 'hello'
RTRIM('hello  ')  -- 'hello'

-- 替换
REPLACE('Hello World', 'World', 'MySQL')  -- 'Hello MySQL'
```

### 数值函数

```sql
-- 四舍五入
ROUND(3.1415, 2)  -- 3.14

-- 向上取整
CEIL(3.14)  -- 4

-- 向下取整
FLOOR(3.14)  -- 3

-- 绝对值
ABS(-5)  -- 5

-- 随机数
RAND()  -- 0 ~ 1 之间的随机数

-- 最大最小值
GREATEST(1, 2, 3)  -- 3
LEAST(1, 2, 3)  -- 1
```

### 日期时间函数

```sql
-- 当前日期时间
NOW()  -- 2025-12-10 19:54:27
CURDATE()  -- 2025-12-10
CURTIME()  -- 19:54:27

-- 日期格式化
DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s')  -- 2025-12-10 19:54:27

-- 提取日期部分
YEAR(NOW())  -- 2025
MONTH(NOW())  -- 12
DAY(NOW())  -- 10
HOUR(NOW())  -- 19

-- 日期计算
DATE_ADD(NOW(), INTERVAL 1 DAY)  -- 明天
DATE_SUB(NOW(), INTERVAL 1 MONTH)  -- 上个月
DATEDIFF('2025-12-10', '2025-12-01')  -- 9

-- 时间戳
UNIX_TIMESTAMP()  -- 当前时间戳
FROM_UNIXTIME(1733832867)  -- 转换时间戳为日期时间
```

### 聚合函数

```sql
-- 计数
COUNT(*)  -- 总行数
COUNT(DISTINCT column)  -- 去重计数

-- 求和
SUM(amount)

-- 平均值
AVG(score)

-- 最大最小值
MAX(price)
MIN(price)

-- 分组拼接
GROUP_CONCAT(name SEPARATOR ', ')
```

### 条件函数

```sql
-- IF 函数
IF(age >= 18, '成年', '未成年')

-- CASE WHEN
CASE
    WHEN age < 18 THEN '未成年'
    WHEN age < 60 THEN '成年'
    ELSE '老年'
END

-- IFNULL
IFNULL(email, 'N/A')  -- email 为 NULL 时返回 'N/A'

-- COALESCE
COALESCE(email, phone, 'N/A')  -- 返回第一个非 NULL 值
```

## 配置参数速查

### 连接相关

```sql
-- 最大连接数
max_connections = 1000

-- 连接超时（秒）
wait_timeout = 28800
interactive_timeout = 28800

-- 最大允许包大小
max_allowed_packet = 64M
```

### InnoDB 相关

```sql
-- 缓冲池大小（最重要的参数）
innodb_buffer_pool_size = 8G

-- 缓冲池实例数
innodb_buffer_pool_instances = 8

-- 日志文件大小
innodb_log_file_size = 1G

-- 日志缓冲区大小
innodb_log_buffer_size = 16M

-- 刷新日志策略
innodb_flush_log_at_trx_commit = 1  -- 1=最安全，2=性能好
```

### 查询相关

```sql
-- 慢查询时间（秒）
long_query_time = 2

-- 慢查询日志
slow_query_log = ON
slow_query_log_file = /var/log/mysql/slow.log

-- 查询缓存（MySQL 8.0 已移除）
query_cache_size = 256M
query_cache_type = 1
```

## SQL 优化速查

### EXPLAIN 执行计划

```sql
EXPLAIN SELECT * FROM users WHERE id = 1;
```

**type 类型（性能从好到差）**:

- `system` - 系统表，只有一行
- `const` - 主键或唯一索引等值查询
- `eq_ref` - 唯一索引扫描
- `ref` - 非唯一索引扫描
- `range` - 范围查询
- `index` - 全索引扫描
- `ALL` - 全表扫描（最差）

### 索引优化技巧

```sql
-- ✅ 使用索引
WHERE name = '张三'
WHERE age > 18
WHERE name LIKE '张%'

-- ❌ 索引失效
WHERE YEAR(created_at) = 2025  -- 函数
WHERE name LIKE '%张三'  -- 前缀模糊
WHERE age != 18  -- 不等于
WHERE name IS NOT NULL  -- IS NOT NULL
```

### 查询优化技巧

```sql
-- ✅ 只查询需要的字段
SELECT id, name FROM users;

-- ❌ 避免 SELECT *
SELECT * FROM users;

-- ✅ 使用 LIMIT
SELECT * FROM users LIMIT 100;

-- ✅ 使用 JOIN 代替子查询
SELECT u.* FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- ❌ 避免子查询
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders);
```

## 常用运维命令

### 查看服务状态

```sql
-- 查看服务器状态
SHOW STATUS;

-- 查看连接
SHOW PROCESSLIST;
SHOW FULL PROCESSLIST;

-- 查看 InnoDB 状态
SHOW ENGINE INNODB STATUS\G

-- 查看表状态
SHOW TABLE STATUS FROM dbname;
```

### 备份恢复

```bash
# 备份单个数据库
mysqldump -u root -p dbname > backup.sql

# 备份所有数据库
mysqldump -u root -p --all-databases > all_backup.sql

# 备份表结构
mysqldump -u root -p --no-data dbname > schema.sql

# 恢复数据库
mysql -u root -p dbname < backup.sql
```

### 主从复制

```sql
-- 主库：查看状态
SHOW MASTER STATUS;

-- 从库：配置主库
CHANGE MASTER TO
    MASTER_HOST='192.168.1.1',
    MASTER_USER='repl',
    MASTER_PASSWORD='password',
    MASTER_LOG_FILE='mysql-bin.000001',
    MASTER_LOG_POS=154;

-- 从库：启动/停止复制
START SLAVE;
STOP SLAVE;

-- 从库：查看状态
SHOW SLAVE STATUS\G
```

## 常用配置文件位置

### Linux

```bash
# 配置文件
/etc/my.cnf
/etc/mysql/my.cnf

# 数据目录
/var/lib/mysql

# 日志文件
/var/log/mysql/error.log
/var/log/mysql/slow.log
```

### macOS

```bash
# 配置文件
/usr/local/etc/my.cnf
/opt/homebrew/etc/my.cnf

# 数据目录
/usr/local/var/mysql
```

### Windows

```
# 配置文件
C:\ProgramData\MySQL\MySQL Server 8.0\my.ini

# 数据目录
C:\ProgramData\MySQL\MySQL Server 8.0\Data
```

## 快速故障排查

### 连接问题

```sql
-- 检查用户权限
SELECT user, host FROM mysql.user;

-- 检查连接数
SHOW STATUS LIKE 'Threads_connected';
SHOW VARIABLES LIKE 'max_connections';
```

### 性能问题

```sql
-- 查看慢查询数量
SHOW GLOBAL STATUS LIKE 'Slow_queries';

-- 查看 QPS
SHOW GLOBAL STATUS LIKE 'Questions';

-- 查看缓冲池命中率
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read%';
```

### 锁问题

```sql
-- 查看锁等待
SHOW ENGINE INNODB STATUS\G

-- 查看事务
SELECT * FROM information_schema.INNODB_TRX;

-- 查看锁
SELECT * FROM performance_schema.data_locks;
```

## 总结

本快速参考手册涵盖了：

- ✅ 常用 SQL 命令速查
- ✅ 数据类型对照表
- ✅ 常用函数列表
- ✅ 配置参数说明
- ✅ SQL 优化技巧
- ✅ 运维命令集合

继续学习 [最佳实践](./best-practices) 和 [常见问题](./faq)！
