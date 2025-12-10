---
sidebar_position: 11
title: MySQL 面试题集
---

# MySQL 面试题集

> [!TIP] > **面试必备**: 本文精选 MySQL 常见面试题，涵盖基础概念、索引优化、事务处理、性能调优等核心知识点。

## 基础概念题

### 1. MySQL 有哪些存储引擎？各有什么特点？

**答案**:

| 存储引擎 | 事务   | 锁粒度 | 外键   | 适用场景            |
| -------- | ------ | ------ | ------ | ------------------- |
| InnoDB   | 支持   | 行锁   | 支持   | 通用 OLTP，默认引擎 |
| MyISAM   | 不支持 | 表锁   | 不支持 | 只读或读多写少      |
| Memory   | 不支持 | 表锁   | 不支持 | 临时表、缓存        |

**重点**: InnoDB 是 MySQL 5.5+ 的默认引擎，支持事务、行锁、外键和 MVCC。

### 2. 什么是 ACID？

**答案**:

- **Atomicity (原子性)** - 事务是不可分割的最小单位，要么全部成功，要么全部失败
- **Consistency (一致性)** - 事务前后数据保持一致状态
- **Isolation (隔离性)** - 并发事务之间相互隔离，互不干扰
- **Durability (持久性)** - 事务提交后，修改永久保存

### 3. MySQL 的四种隔离级别是什么？

**答案**:

| 隔离级别         | 脏读   | 不可重复读 | 幻读   |
| ---------------- | ------ | ---------- | ------ |
| READ UNCOMMITTED | 可能   | 可能       | 可能   |
| READ COMMITTED   | 不可能 | 可能       | 可能   |
| REPEATABLE READ  | 不可能 | 不可能     | 可能\* |
| SERIALIZABLE     | 不可能 | 不可能     | 不可能 |

**注**: MySQL InnoDB 在 REPEATABLE READ 级别通过 MVCC 和 Next-Key Lock 解决了幻读问题。

### 4. 什么是 MVCC？

**答案**:

MVCC (Multi-Version Concurrency Control) 是多版本并发控制，通过保存数据的多个版本实现非锁定读，提高并发性能。

- InnoDB 为每行添加隐藏列：事务 ID、回滚指针
- 事务读取时根据 Read View 判断数据可见性
- 不需要加锁，避免锁等待
- 只在 READ COMMITTED 和 REPEATABLE READ 隔离级别下生效

## 索引相关题

### 5. 什么是索引？索引的优缺点？

**答案**:

**定义**: 索引是帮助 MySQL 高效获取数据的数据结构，类似于书的目录。

**优点**:

- 大幅提升查询速度
- 加速表连接
- 减少分组和排序时间

**缺点**:

- 占用磁盘空间
- 降低写入速度（INSERT、UPDATE、DELETE）
- 需要维护成本

### 6. MySQL 索引使用的是什么数据结构？为什么？

**答案**:

MySQL InnoDB 使用 **B+Tree** 数据结构。

**原因**:

- 所有数据都在叶子节点，范围查询效率高
- 叶子节点之间有指针，方便顺序遍历
- 树的高度较低（通常 3-4 层），减少磁盘 IO
- 支持范围查询、排序、分组

**为什么不用 Hash**:

- Hash 索引不支持范围查询
- Hash 索引不支持排序
- Hash 索引不支持最左前缀匹配

### 7. 什么是聚簇索引和非聚簇索引？

**答案**:

**聚簇索引 (Clustered Index)**:

- 数据和索引存储在一起
- InnoDB 的主键索引就是聚簇索引
- 每张表只能有一个聚簇索引
- 叶子节点存储完整的行数据

**非聚簇索引 (Secondary Index)**:

- 数据和索引分开存储
- InnoDB 的二级索引是非聚簇索引
- 叶子节点存储主键值，需要回表查询

### 8. 什么是覆盖索引？

**答案**:

覆盖索引是指查询的列都在索引中，无需回表查询。

```sql
-- 创建覆盖索引
CREATE INDEX idx_username_email ON users(username, email);

-- 使用覆盖索引（无需回表）
SELECT username, email FROM users WHERE username = '张三';

-- 非覆盖索引（需要回表）
SELECT username, email, age FROM users WHERE username = '张三';
```

### 9. 什么是最左前缀原则？

**答案**:

复合索引 `(a, b, c)` 可以当作 `(a)`、`(a,b)`、`(a,b,c)` 使用，但不能跳过中间列。

```sql
CREATE INDEX idx_abc ON table(a, b, c);

-- ✅ 使用索引
WHERE a = 1
WHERE a = 1 AND b = 2
WHERE a = 1 AND b = 2 AND c = 3

-- ❌ 不使用索引
WHERE b = 2
WHERE c = 3
WHERE b = 2 AND c = 3
```

### 10. 哪些情况会导致索引失效？

**答案**:

1. 在索引列上使用函数
2. 使用 OR 连接（未建索引的列）
3. 类型不匹配（隐式转换）
4. LIKE 以 % 开头
5. 使用 != 或 `<>`
6. IS NOT NULL

```sql
-- ❌ 索引失效
SELECT * FROM users WHERE YEAR(created_at) = 2025;
SELECT * FROM users WHERE username = 123;  -- 类型不匹配
SELECT * FROM users WHERE username LIKE '%张三';

-- ✅ 索引有效
SELECT * FROM users WHERE created_at >= '2025-01-01';
SELECT * FROM users WHERE username = '123';
SELECT * FROM users WHERE username LIKE '张三%';
```

## 事务和锁相关题

### 11. MySQL 有哪些锁？

**答案**:

**按锁粒度分类**:

- 表锁 - 锁定整张表
- 行锁 - 锁定特定行（InnoDB）
- 页锁 - 锁定数据页

**按锁模式分类**:

- 共享锁 (S 锁) - 读锁，多个事务可同时持有
- 排它锁 (X 锁) - 写锁，独占访问

**InnoDB 特有**:

- 间隙锁 (Gap Lock) - 锁定索引间隙
- Next-Key Lock - 行锁 + 间隙锁
- 意向锁 - 表级锁，辅助行锁

### 12. 什么是死锁？如何避免？

**答案**:

**定义**: 两个或多个事务互相等待对方释放锁，形成循环等待。

**避免方法**:

1. 按相同顺序访问表和行
2. 保持事务简短
3. 使用索引访问数据（减少锁范围）
4. 使用较低的隔离级别
5. 合理设置锁等待超时

```sql
-- 设置锁等待超时
SET innodb_lock_wait_timeout = 50;
```

### 13. 什么是幻读？InnoDB 如何解决幻读？

**答案**:

**幻读**: 同一事务中，多次查询记录数不同。

```sql
-- 事务A
SELECT COUNT(*) FROM users WHERE age > 18;  -- 100

-- 事务B插入了一条数据
INSERT INTO users (age) VALUES (20);
COMMIT;

-- 事务A再次查询
SELECT COUNT(*) FROM users WHERE age > 18;  -- 101（幻读）
```

**InnoDB 解决方案**:

- 在 REPEATABLE READ 隔离级别下
- 使用 **Next-Key Lock**（行锁 + 间隙锁）
- 锁定记录和记录之间的间隙，防止插入新记录

## 性能优化题

### 14. 如何优化慢查询？

**答案**:

1. **使用 EXPLAIN 分析执行计划**

```sql
EXPLAIN SELECT * FROM users WHERE age > 18;
```

2. **添加合适的索引**

```sql
CREATE INDEX idx_age ON users(age);
```

3. **避免 SELECT \***

```sql
-- ❌ 不推荐
SELECT * FROM users;

-- ✅ 推荐
SELECT id, username, email FROM users;
```

4. **优化 WHERE 条件**

```sql
-- ❌ 索引失效
WHERE DATE(created_at) = '2025-12-09'

-- ✅ 改写为范围查询
WHERE created_at >= '2025-12-09 00:00:00'
  AND created_at < '2025-12-10 00:00:00'
```

5. **优化分页查询**

```sql
-- ❌ 深分页慢
SELECT * FROM users ORDER BY id LIMIT 100000, 10;

-- ✅ 使用上次最大 ID
SELECT * FROM users WHERE id > 100000 ORDER BY id LIMIT 10;
```

### 15. 什么是回表查询？如何避免？

**答案**:

**回表查询**: 使用二级索引查询时，需要根据主键再次查询完整的行数据。

```sql
-- 创建索引
CREATE INDEX idx_username ON users(username);

-- 回表查询流程
SELECT * FROM users WHERE username = '张三';
-- 1. 先在 idx_username 索引中找到主键 id
-- 2. 再根据 id 在聚簇索引中找到完整行数据（回表）
```

**避免方法 - 使用覆盖索引**:

```sql
-- 创建覆盖索引
CREATE INDEX idx_username_email ON users(username, email);

-- 无需回表
SELECT username, email FROM users WHERE username = '张三';
```

### 16. 主键使用自增 ID 和 UUID 有什么区别？

**答案**:

| 特性     | 自增 ID    | UUID       |
| -------- | ---------- | ---------- |
| 长度     | 8 字节     | 16 字节    |
| 顺序性   | 有序       | 无序       |
| 性能     | 高         | 低         |
| 页分裂   | 少         | 多         |
| 全局唯一 | 单表唯一   | 全局唯一   |
| 适用场景 | 单机数据库 | 分布式系统 |

**推荐**:

- 单机环境用自增 ID
- 分布式环境用雪花算法（类似自增但全局唯一）

## 高级题

### 17. 什么是分库分表？什么时候需要？

**答案**:

**垂直分表**: 将一张表的列拆分到多张表
**水平分表**: 将一张表的行拆分到多张表
**分库**: 将不同业务的表分到不同数据库

**使用场景**:

- 单表数据量过大（> 1000 万）
- 并发量过大
- 磁盘空间不足

### 18. 什么是主从复制？有什么作用？

**答案**:

**原理**: 主库记录 binlog，从库读取 binlog 并重放，保持数据一致。

**作用**:

- 读写分离（主库写，从库读）
- 数据备份
- 高可用（主库故障时切换到从库）

**配置**:

```sql
-- 主库
SHOW MASTER STATUS;

-- 从库
CHANGE MASTER TO
    MASTER_HOST='master_host',
    MASTER_USER='repl_user',
    MASTER_PASSWORD='password';
START SLAVE;
```

### 19. B 树和 B+树的区别？

**答案**:

| 特性     | B 树               | B+树                   |
| -------- | ------------------ | ---------------------- |
| 数据存储 | 所有节点都存储数据 | 只有叶子节点存储数据   |
| 叶子节点 | 无指针连接         | 有指针连接             |
| 范围查询 | 需要中序遍历       | 只需遍历叶子节点       |
| 查询性能 | 不稳定             | 稳定（都要到叶子节点） |

**MySQL 使用 B+树的原因**:

- 范围查询效率高
- 所有查询都要到叶子节点，性能稳定
- 非叶子节点不存数据，可以存更多的索引，降低树高度

### 20. MySQL 的执行计划 type 类型有哪些？性能如何？

**答案**:

从好到差排序：

1. **system** - 表只有一行记录
2. **const** - 通过主键或唯一索引等值查询
3. **eq_ref** - 唯一索引扫描，每次只返回一条记录
4. **ref** - 非唯一索引扫描
5. **range** - 范围查询
6. **index** - 全索引扫描
7. **ALL** - 全表扫描（最差）

**优化目标**: 至少达到 range 级别，最好是 ref 或 const。

## 实战经验题

### 21. 如何定位慢查询？

**答案**:

1. **开启慢查询日志**

```sql
SET GLOBAL slow_query_log = ON;
SET GLOBAL long_query_time = 2;
```

2. **分析慢查询日志**

```bash
mysqldumpslow -s t -t 10 /var/log/mysql/slow.log
```

3. **使用 EXPLAIN 分析**

```sql
EXPLAIN SELECT * FROM users WHERE age > 18;
```

4. **使用性能监控工具**

- MySQL Workbench
- Percona Toolkit
- MySQL Enterprise Monitor

### 22. 生产环境中如何安全地添加索引？

**答案**:

1. **测试环境验证**

- 先在测试环境创建索引
- 验证性能提升效果

2. **使用 ALGORITHM 选项**

```sql
-- 在线添加索引（不锁表）
ALTER TABLE users ADD INDEX idx_age (age) ALGORITHM=INPLACE, LOCK=NONE;
```

3. **选择业务低峰期**

- 凌晨或业务量少的时候操作

4. **监控服务器资源**

- 监控 CPU、内存、磁盘 IO

## 总结

本文涵盖了 MySQL 常见面试题：

- ✅ 基础概念：存储引擎、ACID、隔离级别、MVCC
- ✅ 索引相关：B+树、聚簇索引、覆盖索引、最左前缀
- ✅ 事务和锁：锁类型、死锁、幻读
- ✅ 性能优化：慢查询、回表、分页优化
- ✅ 高级主题：分库分表、主从复制

建议结合 [基础概念](/docs/mysql/basic-concepts)、[索引优化](/docs/mysql/indexes)、[事务处理](/docs/mysql/transactions) 等文档深入学习！
