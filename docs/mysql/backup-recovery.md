---
sidebar_position: 10
title: 备份与恢复
---

# MySQL 备份与恢复

> [!TIP] > **数据安全第一**: 定期备份是保护数据安全的最后一道防线。本文介绍 MySQL 的备份策略和恢复方法。

## 备份类型

### 逻辑备份 vs 物理备份

| 类型 | 逻辑备份      | 物理备份              |
| ---- | ------------- | --------------------- |
| 原理 | 导出 SQL 语句 | 复制数据文件          |
| 工具 | mysqldump     | cp、rsync、xtrabackup |
| 优点 | 跨平台、可读  | 速度快                |
| 缺点 | 速度慢        | 不可读、平台相关      |
| 适用 | 小中型数据库  | 大型数据库            |

### 全量备份 vs 增量备份

- **全量备份** - 备份所有数据
- **增量备份** - 只备份变化的数据
- **差异备份** - 备份自上次全量备份后的变化

## mysqldump 逻辑备份

### 基本使用

```bash
# 备份单个数据库
mysqldump -u root -p database_name > backup.sql

# 备份多个数据库
mysqldump -u root -p --databases db1 db2 > backup.sql

# 备份所有数据库
mysqldump -u root -p --all-databases > all_databases.sql

# 备份单张表
mysqldump -u root -p database_name table_name > table_backup.sql
```

### 常用选项

```bash
# 完整备份（推荐）
mysqldump -u root -p \
    --single-transaction \      # InnoDB 一致性备份
    --routines \                # 包含存储过程和函数
    --triggers \                # 包含触发器
    --events \                  # 包含事件
    --hex-blob \                # 二进制数据十六进制
    --master-data=2 \           # 记录binlog位置
    database_name > backup.sql

# 只备份结构
mysqldump -u root -p --no-data database_name > schema.sql

# 只备份数据
mysqldump -u root -p --no-create-info database_name > data.sql

# 压缩备份
mysqldump -u root -p database_name | gzip > backup.sql.gz
```

### 恢复数据

```bash
# 恢复数据库
mysql -u root -p database_name < backup.sql

# 恢复压缩备份
gunzip < backup.sql.gz | mysql -u root -p database_name

# 在 MySQL 中执行
mysql -u root -p
SOURCE /path/to/backup.sql;
```

## 二进制日志备份

### 开启 binlog

```sql
-- 配置文件 my.cnf
[mysqld]
log-bin=mysql-bin
binlog-format=ROW
expire_logs_days=7
```

### 管理 binlog

```sql
-- 查看 binlog 文件
SHOW BINARY LOGS;

-- 查看当前使用的 binlog
SHOW MASTER STATUS;

-- 查看 binlog 内容
SHOW BINLOG EVENTS IN 'mysql-bin.000001';

-- 删除旧的 binlog
PURGE BINARY LOGS BEFORE '2025-12-01 00:00:00';
PURGE BINARY LOGS TO 'mysql-bin.000010';

-- 刷新 binlog（生成新文件）
FLUSH LOGS;
```

### 使用 binlog 恢复

```bash
# 恢复 binlog
mysqlbinlog mysql-bin.000001 | mysql -u root  -p

# 按时间范围恢复
mysqlbinlog --start-datetime="2025-12-09 10:00:00" \
            --stop-datetime="2025-12-09 12:00:00" \
            mysql-bin.000001 | mysql -u root -p

# 按位置恢复
mysqlbinlog --start-position=1000 \
            --stop-position=2000 \
            mysql-bin.000001 | mysql -u root -p
```

## 物理备份

### 文件复制

```bash
# 停止 MySQL
systemctl stop mysql

# 复制数据目录
cp -r /var/lib/mysql /backup/mysql_backup

# 启动 MySQL
systemctl start mysql
```

### Percona XtraBackup

```bash
# 安装 XtraBackup
apt-get install percona-xtrabackup-80

# 全量备份
xtrabackup --backup --target-dir=/backup/full

# 增量备份
xtrabackup --backup --target-dir=/backup/inc1 \
           --incremental-basedir=/backup/full

# 准备备份
xtrabackup --prepare --target-dir=/backup/full

# 恢复备份
systemctl stop mysql
rm -rf /var/lib/mysql/*
xtrabackup --copy-back --target-dir=/backup/full
chown -R mysql:mysql /var/lib/mysql
systemctl start mysql
```

## 主从复制备份

### 配置主从复制

```sql
-- 主库配置 (my.cnf)
[mysqld]
server-id=1
log-bin=mysql-bin
binlog-format=ROW

-- 创建复制用户
CREATE USER 'repl'@'%' IDENTIFIED BY 'password';
GRANT REPLICATION SLAVE ON *.* TO 'repl'@'%';

-- 查看主库状态
SHOW MASTER STATUS;

-- 从库配置 (my.cnf)
[mysqld]
server-id=2
relay-log=mysql-relay-bin

-- 配置主从关系
CHANGE MASTER TO
    MASTER_HOST='192.168.1.100',
    MASTER_USER='repl',
    MASTER_PASSWORD='password',
    MASTER_LOG_FILE='mysql-bin.000001',
    MASTER_LOG_POS=154;

-- 启动从库
START SLAVE;

-- 查看从库状态
SHOW SLAVE STATUS\G
```

## 备份策略

### 备份计划示例

```bash
# 每日全量备份（凌晨2点）
0 2 * * * /usr/local/bin/mysql_backup.sh

# 备份脚本
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR=/backup/mysql
DATABASE=mydb

# 创建备份目录
mkdir -p $BACKUP_DIR

# 执行备份
mysqldump -u root -p'password' \
    --single-transaction \
    --routines \
    --triggers \
    --events \
    $DATABASE | gzip > $BACKUP_DIR/backup_${DATABASE}_${DATE}.sql.gz

# 删除7天前的备份
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
```

### 备份验证

```bash
# 定期验证备份可用性
gunzip < backup.sql.gz | mysql -u root -p test_restore

# 检查备份文件
mysqldump --fix-backup backup.sql
```

## 灾难恢复

### 恢复流程

1. **全量恢复** - 从最近的全量备份恢复
2. **binlog 恢复** - 应用 binlog 到故障时间点
3. **验证数据** - 检查数据完整性

```bash
# 1. 恢复全量备份
mysql -u root -p < full_backup.sql

# 2. 应用 binlog（恢复到故障前）
mysqlbinlog --start-datetime="2025-12-09 00:00:00" \
            --stop-datetime="2025-12-09 11:59:59" \
            mysql-bin.* | mysql -u root -p

# 3. 验证数据
mysql -u root -p -e "SELECT COUNT(*) FROM database.table"
```

### 误删数据恢复

```bash
# 场景：误删除了数据

# 1. 找到删除操作的 binlog 位置
mysqlbinlog --base64-output=decode-rows -v mysql-bin.000001 | grep DELETE

# 2. 恢复到删除前的位置
mysqlbinlog --stop-position=12345 mysql-bin.000001 | mysql -u root -p

# 3. 跳过删除操作，继续后续操作
mysqlbinlog --start-position=12346 mysql-bin.000001 | mysql -u root -p
```

## 最佳实践

> [!IMPORTANT] > **备份最佳实践**:
>
> 1. ✅ 定期自动备份（每日/每周）
> 2. ✅ 异地存储备份文件
> 3. ✅ 定期验证备份可用性
> 4. ✅ 开启 binlog 用于增量恢复
> 5. ✅ 记录备份和恢复流程
> 6. ✅ 测试恢复流程
> 7. ✅ 监控备份任务执行状态

### 3-2-1 备份原则

- **3** 份数据副本
- **2** 种不同的存储介质
- **1** 份异地备份

## 总结

本文介绍了 MySQL 备份与恢复：

- ✅ 备份类型：逻辑备份、物理备份
- ✅ mysqldump 备份和恢复
- ✅ binlog 增量备份
- ✅ 主从复制
- ✅ 备份策略和验证
- ✅ 灾难恢复流程

最后学习 [面试题集](/docs/interview/mysql-interview-questions)！
