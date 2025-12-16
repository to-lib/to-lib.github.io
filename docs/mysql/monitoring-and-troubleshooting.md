---
sidebar_position: 15
title: 监控与排障
---

# MySQL 监控与排障

> [!TIP] > **核心思路**：先确认“是不是 MySQL 的问题”，再定位是 **资源**（CPU/IO/内存/磁盘）、**查询**（慢 SQL/锁等待）、还是 **架构**（复制延迟/连接池）导致。

## 你需要关注的信号

- **延迟**：P95/P99 响应时间上升
- **吞吐**：QPS/TPS 异常波动
- **连接数**：`Threads_connected` 持续逼近 `max_connections`
- **慢查询**：`Slow_queries` 增加、慢日志堆积
- **锁等待**：事务堆积、死锁
- **复制**：`Seconds_Behind_Master` 增大或 IO/SQL 线程不运行

## 常用诊断命令

### 连接与会话

```sql
SHOW PROCESSLIST;
SHOW FULL PROCESSLIST;

SHOW STATUS LIKE 'Threads_connected';
SHOW VARIABLES LIKE 'max_connections';
```

### 引擎状态

```sql
SHOW ENGINE INNODB STATUS\G
```

### 性能视图（MySQL 8.0 推荐）

```sql
SELECT * FROM performance_schema.data_lock_waits;
SELECT * FROM performance_schema.data_locks;
```

如果启用 `sys` 库，也可以用更友好的视图：

```sql
SELECT * FROM sys.schema_table_lock_waits;
SELECT * FROM sys.schema_unused_indexes;
```

## 慢查询定位

### 开启慢查询日志

```sql
SET GLOBAL slow_query_log = ON;
SET GLOBAL long_query_time = 1;
SHOW VARIABLES LIKE 'slow_query_log_file';
```

常见分析方式：

- 先按耗时/次数聚合
- 再对 Top SQL 做 `EXPLAIN`

相关优化建议可参考：[/docs/mysql/performance-optimization](/docs/mysql/performance-optimization) 与 [/docs/mysql/indexes](/docs/mysql/indexes)

## 锁等待与死锁

### 常用排查路径

- 是否存在长事务（事务开启后很久不提交）
- 是否未走索引导致锁范围扩大
- 是否访问顺序不一致导致死锁

死锁信息：

```sql
SHOW ENGINE INNODB STATUS\G
```

并发控制与锁机制详细说明：[/docs/mysql/transactions](/docs/mysql/transactions) 与 [/docs/mysql/locks](/docs/mysql/locks)

## 复制延迟排查（如果启用了主从）

```sql
SHOW SLAVE STATUS\G
```

重点字段：

- `Slave_IO_Running`
- `Slave_SQL_Running`
- `Seconds_Behind_Master`
- `Last_IO_Error` / `Last_SQL_Error`

复制原理与优化：[/docs/mysql/replication](/docs/mysql/replication)

## 常见故障与处理思路

### 1) 连接打满/应用报“Too many connections”

- 增大连接池复用，避免每次请求都建连
- 排查是否有连接泄漏（长时间 `Sleep`）
- 临时措施：适度提高 `max_connections`（要评估内存）

```sql
SHOW PROCESSLIST;
SHOW VARIABLES LIKE 'wait_timeout';
```

### 2) 磁盘写满

- 清理历史 binlog/慢日志/中间文件
- 调整 binlog 保留策略

```sql
SHOW BINARY LOGS;
PURGE BINARY LOGS BEFORE '2025-01-01 00:00:00';
```

### 3) CPU 飙高但 QPS 不高

- 可能是某些查询在做大量计算/排序/临时表
- 可能是锁等待引发大量上下文切换
- 通过慢日志/执行计划定位 Top SQL

### 4) IO 飙高/响应变慢

- 检查是否在大量全表扫描
- 检查 Buffer Pool 命中率

```sql
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read%';
```

### 5) 误操作数据恢复

- 优先使用“备份 + binlog 定点恢复”

参考：[/docs/mysql/backup-recovery](/docs/mysql/backup-recovery)

## 建议的日常监控指标

- QPS/TPS
- 慢查询数量与分位耗时
- 连接数与连接创建速率
- InnoDB Buffer Pool 命中率
- Redo/Undo 压力（间接表现：checkpoint/flush 频繁）
- 主从复制延迟
- 磁盘空间与增长速度

## 下一步

- 性能优化：[/docs/mysql/performance-optimization](/docs/mysql/performance-optimization)
- 最佳实践：[/docs/mysql/best-practices](/docs/mysql/best-practices)
- 常见问题：[/docs/mysql/faq](/docs/mysql/faq)
