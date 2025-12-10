---
sidebar_position: 8
title: 备份与恢复
---

# PostgreSQL 备份与恢复

数据备份是保障数据安全的重要措施。本文介绍 PostgreSQL 的备份和恢复策略。

## 📦 备份类型

### 1. 逻辑备份

导出 SQL 语句或数据文件。

**优点：**

- 可读性强
- 可跨版本恢复
- 可选择性备份

**缺点：**

- 备份/恢复慢
- 文件较大

### 2. 物理备份

直接复制数据文件。

**优点：**

- 速度快
- 文件较小

**缺点：**

- 不可跨版本
- 必须完整备份

## 🛠️ 逻辑备份工具

### pg_dump - 备份单个数据库

```bash
# 基本备份
pg_dump mydb > mydb_backup.sql

# 指定主机和用户
pg_dump -h localhost -U postgres mydb > backup.sql

# 自定义格式（压缩，推荐）
pg_dump -Fc mydb > mydb_backup.dump

# 目录格式（并行备份）
pg_dump -Fd mydb -f mydb_backup_dir -j 4

# 仅备份数据（不包括结构）
pg_dump --data-only mydb > data.sql

# 仅备份结构
pg_dump --schema-only mydb > schema.sql

# 备份特定表
pg_dump -t users -t orders mydb > tables.sql
```

### pg_dumpall - 备份所有数据库

```bash
# 备份所有数据库
pg_dumpall > all_databases.sql

# 仅备份全局对象（角色、表空间）
pg_dumpall --globals-only > globals.sql

# 仅备份角色
pg_dumpall --roles-only > roles.sql
```

## 🔄 恢复数据

### psql - 恢复 SQL 文件

```bash
# 恢复 SQL 备份
psql mydb < mydb_backup.sql

# 创建数据库并恢复
createdb mydb
psql mydb < mydb_backup.sql

# 指定主机和用户
psql -h localhost -U postgres mydb < backup.sql
```

### pg_restore - 恢复自定义格式

```bash
# 恢复自定义格式备份
pg_restore -d mydb mydb_backup.dump

# 并行恢复
pg_restore -d mydb -j 4 mydb_backup.dump

# 仅恢复数据
pg_restore --data-only -d mydb backup.dump

# 仅恢复特定表
pg_restore -t users -d mydb backup.dump

# 先清理再恢复
pg_restore --clean -d mydb backup.dump
```

## 💾 物理备份

### 1. 文件系统级备份

```bash
# 停止 PostgreSQL
sudo systemctl stop postgresql

# 复制数据目录
sudo cp -r /var/lib/postgresql/15/main /backup/pg_data_backup

# 启动 PostgreSQL
sudo systemctl start postgresql
```

### 2. pg_basebackup - 在线物理备份

```bash
# 基础备份
pg_basebackup -D /backup/pg_base -Ft -z -P

# 参数说明：
# -D: 备份目录
# -Ft: tar 格式
# -z: 压缩
# -P: 显示进度

# 流式备份
pg_basebackup -D /backup/pg_base -Xs -P

# WAL 归档备份
pg_basebackup -D /backup/pg_base -X fetch -P
```

### 恢复物理备份

```bash
# 停止 PostgreSQL
sudo systemctl stop postgresql

# 清空数据目录
sudo rm -rf /var/lib/postgresql/15/main/*

# 解压备份
sudo tar -xzf /backup/base.tar.gz -C /var/lib/postgresql/15/main/

# 修复权限
sudo chown -R postgres:postgres /var/lib/postgresql/15/main

# 启动 PostgreSQL
sudo systemctl start postgresql
```

## 🔁 持续归档和时间点恢复（PITR）

### 1. 配置 WAL 归档

编辑 `postgresql.conf`：

```conf
# 启用归档
wal_level = replica
archive_mode = on
archive_command = 'cp %p /archive/%f'

# 或使用 rsync
archive_command = 'rsync -a %p /backup/archive/%f'
```

重启 PostgreSQL：

```bash
sudo systemctl restart postgresql
```

### 2. 基础备份

```bash
pg_basebackup -D /backup/base -Ft -z -P
```

### 3. 恢复到特定时间点

```bash
# 停止 PostgreSQL
sudo systemctl stop postgresql

# 恢复基础备份
cd /var/lib/postgresql/15/main
sudo tar -xzf /backup/base/base.tar.gz

# 创建 recovery.conf（PostgreSQL 12+使用 recovery.signal）
sudo touch recovery.signal

# 配置恢复参数（postgresql.auto.conf）
restore_command = 'cp /archive/%f %p'
recovery_target_time = '2024-01-15 14:30:00'

# 启动 PostgreSQL
sudo systemctl start postgresql

# 查看恢复状态
sudo tail -f /var/log/postgresql/postgresql-15-main.log
```

## 📅 自动化备份

### Cron 定时任务

```bash
# 编辑 crontab
crontab -e

# 每天凌晨 2 点备份
0 2 * * * /usr/bin/pg_dump -Fc mydb > /backup/mydb_$(date +\%Y\%m\%d).dump

# 清理 7 天前的备份
0 3 * * * find /backup -name "*.dump" -mtime +7 -delete
```

### 备份脚本

```bash
#!/bin/bash

# backup.sh
BACKUP_DIR="/backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="mydb"

# 创建备份
pg_dump -Fc $DB_NAME > $BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.dump

# 检查备份是否成功
if [ $? -eq 0 ]; then
    echo "Backup successful: ${DB_NAME}_${TIMESTAMP}.dump"

    # 清理 7 天前的备份
    find $BACKUP_DIR -name "${DB_NAME}_*.dump" -mtime +7 -delete
else
    echo "Backup failed!" >&2
    exit 1
fi
```

## 🧪 验证备份

### 1. 测试恢复

```bash
# 创建测试数据库
createdb test_restore

# 恢复备份
pg_restore -d test_restore backup.dump

# 验证数据
psql test_restore -c "SELECT COUNT(*) FROM users;"

# 删除测试数据库
dropdb test_restore
```

### 2. 备份完整性检查

```bash
# 检查备份文件
pg_restore --list backup.dump

# 验证备份内容
pg_restore --schema-only backup.dump | less
```

## 💡 最佳实践

### 1. 3-2-1 备份策略

- **3** 份数据副本
- **2** 种不同存储介质
- **1** 份异地备份

### 2. 定期测试恢复

```bash
# 每月测试恢复流程
0 0 1 * * /path/to/test_restore.sh
```

### 3. 监控备份状态

```sql
-- 检查最后一次成功的归档时间
SELECT *
FROM pg_stat_archiver;
```

### 4. 加密备份

```bash
# 使用 GPG 加密
pg_dump mydb | gzip | gpg -e -r your@email.com > backup.sql.gz.gpg

# 解密恢复
gpg -d backup.sql.gz.gpg | gunzip | psql mydb
```

## 📊 备份策略对比

| 策略     | 频率   | 保留期   | 适用场景       |
| -------- | ------ | -------- | -------------- |
| 完整备份 | 每周   | 4 周     | 大型数据库     |
| 增量备份 | 每天   | 7 天     | 变化频繁的数据 |
| WAL 归档 | 实时   | 根据需求 | 需要时间点恢复 |
| 快照     | 每小时 | 24 小时  | 云环境         |

## 🔍 故障恢复场景

### 1. 误删除数据

```sql
-- 使用时间点恢复到删除之前
recovery_target_time = '2024-01-15 13:59:00'
```

### 2. 数据损坏

```bash
# 使用最新的备份恢复
pg_restore -d mydb latest_backup.dump
```

### 3. 硬件故障

```bash
# 在新服务器上恢复
# 1. 安装 PostgreSQL
# 2. 恢复基础备份
# 3. 应用 WAL 日志
```

## 📚 相关资源

- [性能优化](./performance-optimization) - 优化备份性能
- [高可用](./replication) - 主从复制
- [安全管理](./security) - 备份加密

下一节：[快速参考](./quick-reference)
