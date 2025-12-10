---
sidebar_position: 9
title: 快速参考
---

# PostgreSQL 快速参考

常用命令和语法速查表。

## 🔌 连接数据库

```bash
# 连接本地数据库
psql mydb

# 指定用户
psql -U postgres mydb

# 指定主机和端口
psql -h localhost -p 5432 -U postgres mydb

# 执行 SQL 文件
psql -f script.sql mydb

# 在命令行执行 SQL
psql -c "SELECT * FROM users;" mydb
```

## 📊 psql 常用命令

```sql
\l                  -- 列出所有数据库
\c mydb             -- 切换数据库
\dt                 -- 列出当前数据库的所有表
\d users            -- 查看表结构
\di                 -- 列出所有索引
\dv                 -- 列出所有视图
\df                 -- 列出所有函数
\du                 -- 列出所有用户
\dn                 -- 列出所有模式
\timing             -- 开启/关闭查询计时
\x                  -- 切换扩展显示模式
\q                  -- 退出 psql
\h CREATE TABLE     -- 查看 SQL 命令帮助
\?                  -- 查看 psql 命令帮助
```

## 📝 DDL 快速参考

```sql
-- 数据库
CREATE DATABASE mydb;
DROP DATABASE mydb;

-- 表
CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100));
ALTER TABLE users ADD COLUMN age INT;
ALTER TABLE users DROP COLUMN age;
DROP TABLE users;
TRUNCATE TABLE users;

-- 索引
CREATE INDEX idx_name ON users(name);
DROP INDEX idx_name;
REINDEX TABLE users;

-- 视图
CREATE VIEW v_users AS SELECT * FROM users WHERE is_active = true;
DROP VIEW v_users;
```

## 🔍 DML 快速参考

```sql
-- 插入
INSERT INTO users (name) VALUES ('Alice');
INSERT INTO users (name) VALUES ('Bob'), ('Charlie');

-- 查询
SELECT * FROM users;
SELECT name FROM users WHERE age > 25;
SELECT * FROM users ORDER BY age DESC LIMIT 10;

-- 更新
UPDATE users SET age = 30 WHERE name = 'Alice';

-- 删除
DELETE FROM users WHERE id = 1;
```

## 🎯 常用函数

### 字符串函数

```sql
LENGTH('hello')                 -- 5
UPPER('hello')                  -- 'HELLO'
LOWER('HELLO')                  -- 'hello'
CONCAT('hello', ' ', 'world')   -- 'hello world'
SUBSTRING('hello', 1, 3)        -- 'hel'
REPLACE('hello', 'l', 'r')      -- 'herro'
TRIM('  hello  ')               -- 'hello'
```

### 数值函数

```sql
ABS(-5)                         -- 5
ROUND(3.14159, 2)               -- 3.14
CEIL(3.2)                       -- 4
FLOOR(3.8)                      -- 3
POWER(2, 3)                     -- 8
SQRT(16)                        -- 4
```

### 日期时间函数

```sql
NOW()                           -- 当前时间戳
CURRENT_DATE                    -- 当前日期
CURRENT_TIME                    -- 当前时间
AGE('2024-01-01', '2023-01-01') -- 1 年
DATE_PART('year', NOW())        -- 年份
DATE_TRUNC('day', NOW())        -- 截断到天
```

### 聚合函数

```sql
COUNT(*)                        -- 行数
SUM(amount)                     -- 总和
AVG(age)                        -- 平均值
MAX(age)                        -- 最大值
MIN(age)                        -- 最小值
```

## 🔗 JOIN 语法

```sql
-- INNER JOIN
SELECT * FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN
SELECT * FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- RIGHT JOIN
SELECT * FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;

-- FULL OUTER JOIN
SELECT * FROM users u
FULL OUTER JOIN orders o ON u.id = o.user_id;
```

## 📊 窗口函数

```sql
ROW_NUMBER() OVER (ORDER BY age)
RANK() OVER (ORDER BY age)
DENSE_RANK() OVER (ORDER BY age)
LAG(salary) OVER (ORDER BY date)
LEAD(salary) OVER (ORDER BY date)
FIRST_VALUE(name) OVER (PARTITION BY dept ORDER BY salary DESC)
```

## 🎓 CTE（公用表表达式）

```sql
WITH adult_users AS (
    SELECT * FROM users WHERE age >= 18
)
SELECT * FROM adult_users;
```

## 🔐 权限管理

```sql
-- 创建用户
CREATE USER myuser WITH PASSWORD 'password';

-- 授权
GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;
GRANT SELECT ON users TO myuser;
GRANT SELECT, INSERT, UPDATE ON orders TO myuser;

-- 撤销权限
REVOKE ALL ON users FROM myuser;

-- 删除用户
DROP USER myuser;
```

## 📦 备份与恢复

```bash
# 备份
pg_dump mydb > backup.sql
pg_dump -Fc mydb > backup.dump

# 恢复
psql mydb < backup.sql
pg_restore -d mydb backup.dump

# 备份所有数据库
pg_dumpall > all_backup.sql
```

## 🔧 配置查看

```sql
-- 查看配置
SHOW shared_buffers;
SHOW work_mem;
SHOW ALL;

-- 修改配置（会话级别）
SET work_mem = '16MB';

-- 查看数据目录
SHOW data_directory;

-- 查看版本
SELECT version();
```

## 📈 性能监控

```sql
-- 查看活动连接
SELECT * FROM pg_stat_activity;

-- 查看表大小
SELECT pg_size_pretty(pg_total_relation_size('users'));

-- 查看索引使用情况
SELECT * FROM pg_stat_user_indexes WHERE schemaname = 'public';

-- 查看慢查询
SELECT query, mean_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

## 🛠️ 维护命令

```sql
-- VACUUM
VACUUM users;
VACUUM ANALYZE users;

-- REINDEX
REINDEX TABLE users;
REINDEX DATABASE mydb;

-- 更新统计信息
ANALYZE users;
```

## 📋 数据类型速查

| 类型          | 示例                                   |
| ------------- | -------------------------------------- |
| INTEGER       | 123                                    |
| BIGINT        | 9223372036854775807                    |
| NUMERIC(10,2) | 123.45                                 |
| VARCHAR(100)  | 'hello'                                |
| TEXT          | 'long text...'                         |
| BOOLEAN       | true, false                            |
| DATE          | '2024-01-15'                           |
| TIMESTAMP     | '2024-01-15 14:30:00'                  |
| JSON          | `'{"key": "value"}'`                   |
| JSONB         | `'{"key": "value"}'`                   |
| ARRAY         | ARRAY[1,2,3]                           |
| UUID          | 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' |

## 🔍 EXPLAIN 关键词

```sql
EXPLAIN SELECT * FROM users;
EXPLAIN ANALYZE SELECT * FROM users;
EXPLAIN (BUFFERS, ANALYZE) SELECT * FROM users;
```

**关键指标：**

- **Seq Scan**：全表扫描
- **Index Scan**：索引扫描
- **cost**：估算成本
- **rows**：估算行数
- **actual time**：实际时间

## 📚 相关资源

- [基础概念](./basic-concepts)
- [SQL 语法](./sql-syntax)
- [索引优化](./indexes)
