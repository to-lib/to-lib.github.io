---
sidebar_position: 12
title: 主从复制
---

# PostgreSQL 主从复制

主从复制是实现数据库高可用和读写分离的基础。PostgreSQL 提供了流复制（Streaming Replication）和逻辑复制（Logical Replication）两种方式。

## 📚 复制类型对比

| 特性         | 流复制          | 逻辑复制           |
| ------------ | --------------- | ------------------ |
| **复制级别** | 物理级别（WAL） | 逻辑级别（行变更） |
| **版本要求** | 相同主版本      | 可跨版本           |
| **表选择**   | 整个数据库集群  | 可选择特定表       |
| **主从版本** | 必须相同架构    | 可以不同架构       |
| **性能**     | 更高            | 稍低               |
| **适用场景** | 高可用、备份    | 数据分发、升级     |

## 🔄 流复制（Streaming Replication）

流复制是 PostgreSQL 最常用的复制方式，通过传输 WAL 日志实现。

### 1. 主库配置

**postgresql.conf：**

```conf
# 启用 WAL 复制
wal_level = replica

# 允许的最大 WAL 发送进程数
max_wal_senders = 5

# 保留的 WAL 段大小（MB）
wal_keep_size = 1024

# 启用归档（可选，用于 PITR）
archive_mode = on
archive_command = 'cp %p /archive/%f'

# 允许的最大复制槽数
max_replication_slots = 5

# 同步模式（可选）
synchronous_commit = on
synchronous_standby_names = 'standby1'
```

**pg_hba.conf：**

```conf
# 允许从库连接
host    replication     replicator      10.0.0.0/24     md5
```

**创建复制用户：**

```sql
CREATE USER replicator WITH REPLICATION LOGIN PASSWORD 'your_password';
```

**创建复制槽（推荐）：**

```sql
SELECT pg_create_physical_replication_slot('standby1_slot');
```

### 2. 从库配置

**初始化从库：**

```bash
# 使用 pg_basebackup 创建基础备份
pg_basebackup -h master_ip -U replicator -D /var/lib/postgresql/16/main \
    -Fp -Xs -P -R

# -Fp: plain 格式
# -Xs: 流模式传输 WAL
# -P: 显示进度
# -R: 自动创建 standby.signal 和 postgresql.auto.conf
```

**postgresql.conf（从库）：**

```conf
# 从库模式
hot_standby = on

# 从库反馈（帮助主库了解从库状态）
hot_standby_feedback = on

# 主库连接信息（由 pg_basebackup -R 自动创建）
# primary_conninfo = 'host=master_ip port=5432 user=replicator password=xxx'
# primary_slot_name = 'standby1_slot'
```

### 3. 启动从库

```bash
# 确保 standby.signal 文件存在
touch /var/lib/postgresql/16/main/standby.signal

# 启动服务
sudo systemctl start postgresql
```

### 4. 验证复制状态

**在主库上：**

```sql
-- 查看复制状态
SELECT client_addr, state, sent_lsn, write_lsn, flush_lsn, replay_lsn
FROM pg_stat_replication;

-- 查看复制延迟
SELECT client_addr,
       pg_wal_lsn_diff(sent_lsn, replay_lsn) AS replication_lag_bytes
FROM pg_stat_replication;
```

**在从库上：**

```sql
-- 确认是从库
SELECT pg_is_in_recovery();

-- 查看复制状态
SELECT status, sender_host, sender_port
FROM pg_stat_wal_receiver;
```

## 🔀 同步与异步复制

### 异步复制（默认）

- 主库不等待从库确认
- 性能最高
- 可能丢失少量数据

```conf
synchronous_commit = off
```

### 同步复制

- 主库等待从库确认
- 保证数据不丢失
- 性能略低

```conf
synchronous_commit = on
synchronous_standby_names = 'standby1'
```

### 同步模式选项

```conf
# 等待至少 2 个从库确认
synchronous_standby_names = 'FIRST 2 (standby1, standby2, standby3)'

# 等待任意 1 个从库确认
synchronous_standby_names = 'ANY 1 (standby1, standby2)'
```

## 📋 逻辑复制（Logical Replication）

逻辑复制可以选择性地复制特定表，支持跨版本复制。

### 1. 发布者配置（主库）

**postgresql.conf：**

```conf
wal_level = logical
max_replication_slots = 5
max_wal_senders = 5
```

**创建发布：**

```sql
-- 发布所有表
CREATE PUBLICATION my_pub FOR ALL TABLES;

-- 发布特定表
CREATE PUBLICATION my_pub FOR TABLE users, orders;

-- 发布特定操作
CREATE PUBLICATION my_pub FOR TABLE users
    WITH (publish = 'insert, update');
```

### 2. 订阅者配置（从库）

**创建订阅：**

```sql
CREATE SUBSCRIPTION my_sub
    CONNECTION 'host=master_ip port=5432 dbname=mydb user=replicator password=xxx'
    PUBLICATION my_pub;
```

### 3. 管理逻辑复制

```sql
-- 查看发布
SELECT * FROM pg_publication;
SELECT * FROM pg_publication_tables;

-- 查看订阅
SELECT * FROM pg_subscription;
SELECT * FROM pg_stat_subscription;

-- 添加/删除表到发布
ALTER PUBLICATION my_pub ADD TABLE new_table;
ALTER PUBLICATION my_pub DROP TABLE old_table;

-- 刷新订阅
ALTER SUBSCRIPTION my_sub REFRESH PUBLICATION;

-- 禁用/启用订阅
ALTER SUBSCRIPTION my_sub DISABLE;
ALTER SUBSCRIPTION my_sub ENABLE;

-- 删除订阅
DROP SUBSCRIPTION my_sub;
```

## 🔧 故障转移

### 手动故障转移

**在从库上提升为主库：**

```bash
# 方式 1：使用 pg_ctl
pg_ctl promote -D /var/lib/postgresql/16/main

# 方式 2：创建触发文件
touch /var/lib/postgresql/16/main/promote.trigger

# 方式 3：使用 SQL
SELECT pg_promote();
```

### 自动故障转移

推荐使用工具：

- **Patroni**：最流行的自动故障转移方案
- **Repmgr**：老牌的复制管理工具
- **pg_auto_failover**：Citus 提供的方案

## 📊 监控复制

### 监控查询

```sql
-- 复制延迟（秒）
SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds;

-- 复制槽使用情况
SELECT slot_name, slot_type, active, restart_lsn
FROM pg_replication_slots;

-- 未使用的复制槽会阻止 WAL 清理
SELECT slot_name, pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) AS lag_bytes
FROM pg_replication_slots;
```

### 设置报警

```sql
-- 如果延迟超过 60 秒
SELECT CASE
    WHEN EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) > 60
    THEN '复制延迟告警'
    ELSE '正常'
END AS status;
```

## 💡 最佳实践

1. **使用复制槽**：防止 WAL 被过早清理
2. **监控复制延迟**：及时发现问题
3. **测试故障转移**：定期演练
4. **使用同步复制**：对数据安全要求高的场景
5. **配置 hot_standby_feedback**：避免查询冲突

## ⚠️ 常见问题

### 复制槽未激活

```sql
-- 删除不再使用的复制槽
SELECT pg_drop_replication_slot('unused_slot');
```

### 从库查询冲突

```conf
# 增加取消查询前的等待时间
max_standby_streaming_delay = 30s
```

### WAL 目录占用过大

```sql
-- 检查哪个复制槽卡住了
SELECT slot_name, pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) AS lag_bytes
FROM pg_replication_slots
ORDER BY lag_bytes DESC;
```

## 📚 相关资源

- [高可用架构](/docs/postgres/high-availability) - 了解高可用方案
- [备份恢复](/docs/postgres/backup-recovery) - 数据备份策略
- [性能优化](/docs/postgres/performance-optimization) - 优化复制性能
