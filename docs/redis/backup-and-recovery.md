---
sidebar_position: 26
title: 备份与恢复
---

# Redis 备份与恢复

Redis 的备份恢复是数据安全的重要保障。本文详细介绍 RDB 和 AOF 两种持久化文件的备份恢复策略，包括自动化脚本、云存储集成和灾难恢复方案。

## 备份策略概述

### 持久化文件类型

| 文件类型 | 文件名 | 特点 | 适用场景 |
|----------|--------|------|----------|
| RDB | `dump.rdb` | 二进制快照，体积小，恢复快 | 定期备份，冷备份 |
| AOF | `appendonly.aof` | 命令日志，数据完整性高 | 热备份，数据敏感场景 |
| 混合 AOF | `appendonlydir/` | Redis 7.0+，多文件结构 | 现代部署 |

### 备份策略选择

| 场景 | 推荐策略 | 备份频率 |
|------|----------|----------|
| 缓存服务 | 仅 RDB 或不备份 | 每日 |
| 会话存储 | RDB + AOF | 每小时 |
| 关键业务数据 | AOF (everysec) + RDB | 实时 + 每小时 |
| 金融级数据 | AOF (always) + 多副本 | 实时 |

## RDB 备份

### 手动触发备份

```bash
# 后台异步保存（推荐，不阻塞）
redis-cli BGSAVE

# 同步保存（阻塞，仅调试使用）
redis-cli SAVE

# 查看最后保存时间
redis-cli LASTSAVE

# 查看保存状态
redis-cli INFO persistence | grep rdb
```

### 自动备份脚本

**基础备份脚本**：

```bash
#!/bin/bash
# /opt/scripts/redis-backup.sh
# Redis RDB 备份脚本

# 配置
REDIS_HOST="127.0.0.1"
REDIS_PORT="6379"
REDIS_PASSWORD=""
REDIS_DATA_DIR="/var/lib/redis"
BACKUP_DIR="/backup/redis"
RETENTION_DAYS=7

# 日期
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/dump_$DATE.rdb"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 写入日志
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 触发 BGSAVE
log "开始触发 BGSAVE..."
if [ -n "$REDIS_PASSWORD" ]; then
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -a "$REDIS_PASSWORD" BGSAVE
else
    redis-cli -h $REDIS_HOST -p $REDIS_PORT BGSAVE
fi

# 等待 BGSAVE 完成
log "等待 BGSAVE 完成..."
while true; do
    if [ -n "$REDIS_PASSWORD" ]; then
        LASTSAVE=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a "$REDIS_PASSWORD" LASTSAVE 2>/dev/null)
    else
        LASTSAVE=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT LASTSAVE 2>/dev/null)
    fi
    
    if [ "$LASTSAVE" != "$PREV_LASTSAVE" ] && [ -n "$PREV_LASTSAVE" ]; then
        break
    fi
    PREV_LASTSAVE=$LASTSAVE
    sleep 1
done

# 复制 RDB 文件
log "复制 RDB 文件到备份目录..."
cp "$REDIS_DATA_DIR/dump.rdb" "$BACKUP_FILE"

# 验证备份文件
if [ -f "$BACKUP_FILE" ]; then
    SIZE=$(ls -lh "$BACKUP_FILE" | awk '{print $5}')
    log "备份成功: $BACKUP_FILE ($SIZE)"
else
    log "备份失败: 文件不存在"
    exit 1
fi

# 清理过期备份
log "清理 $RETENTION_DAYS 天前的备份..."
find $BACKUP_DIR -name "dump_*.rdb" -mtime +$RETENTION_DAYS -delete

log "备份完成"
```

### 定时任务配置

**Cron 方式**：

```bash
# 编辑 crontab
crontab -e

# 每小时备份一次
0 * * * * /opt/scripts/redis-backup.sh >> /var/log/redis-backup.log 2>&1

# 每天凌晨 2 点备份
0 2 * * * /opt/scripts/redis-backup.sh >> /var/log/redis-backup.log 2>&1

# 每 6 小时备份一次
0 */6 * * * /opt/scripts/redis-backup.sh >> /var/log/redis-backup.log 2>&1
```

**Systemd Timer 方式**：

```ini
# /etc/systemd/system/redis-backup.service
[Unit]
Description=Redis Backup Service
After=redis.service

[Service]
Type=oneshot
ExecStart=/opt/scripts/redis-backup.sh
User=redis
Group=redis

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/redis-backup.timer
[Unit]
Description=Run Redis Backup every hour

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
# 启用定时器
sudo systemctl daemon-reload
sudo systemctl enable --now redis-backup.timer

# 查看定时器状态
sudo systemctl list-timers redis-backup.timer
```

## AOF 备份

### AOF 文件结构

**Redis 6.x 及之前**：

```
/var/lib/redis/
└── appendonly.aof
```

**Redis 7.0+ 多部分 AOF**：

```
/var/lib/redis/appendonlydir/
├── appendonly.aof.1.base.rdb    # 基础 RDB
├── appendonly.aof.2.incr.aof    # 增量 AOF
└── appendonly.aof.manifest      # 清单文件
```

### AOF 备份脚本

```bash
#!/bin/bash
# /opt/scripts/redis-aof-backup.sh
# Redis AOF 备份脚本

REDIS_HOST="127.0.0.1"
REDIS_PORT="6379"
REDIS_DATA_DIR="/var/lib/redis"
BACKUP_DIR="/backup/redis/aof"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# 触发 AOF 重写（压缩文件）
redis-cli -h $REDIS_HOST -p $REDIS_PORT BGREWRITEAOF

# 等待重写完成
echo "等待 AOF 重写完成..."
while true; do
    REWRITING=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT INFO persistence | grep aof_rewrite_in_progress | cut -d: -f2 | tr -d '\r')
    if [ "$REWRITING" = "0" ]; then
        break
    fi
    sleep 2
done

# 备份 AOF 文件
if [ -d "$REDIS_DATA_DIR/appendonlydir" ]; then
    # Redis 7.0+ 多部分 AOF
    tar -czf "$BACKUP_DIR/aof_$DATE.tar.gz" -C "$REDIS_DATA_DIR" appendonlydir
else
    # 传统单文件 AOF
    cp "$REDIS_DATA_DIR/appendonly.aof" "$BACKUP_DIR/appendonly_$DATE.aof"
fi

echo "AOF 备份完成: $BACKUP_DIR"

# 清理 7 天前的备份
find $BACKUP_DIR -name "*.aof" -o -name "*.tar.gz" -mtime +7 -delete
```

### AOF 备份最佳实践

1. **备份前执行 BGREWRITEAOF**：压缩 AOF 文件大小
2. **避免高峰期备份**：AOF 重写会消耗 CPU 和磁盘 I/O
3. **备份整个 appendonlydir 目录**（Redis 7.0+）
4. **验证备份文件完整性**：

```bash
# 检查 AOF 文件
redis-check-aof --fix appendonly.aof

# Redis 7.0+ 检查
redis-check-aof --fix appendonlydir/appendonly.aof.manifest
```

## 云存储备份

### AWS S3 备份

```bash
#!/bin/bash
# /opt/scripts/redis-backup-s3.sh

BACKUP_DIR="/backup/redis"
S3_BUCKET="s3://your-bucket/redis-backup"
DATE=$(date +%Y%m%d_%H%M%S)

# 执行本地备份
/opt/scripts/redis-backup.sh

# 上传到 S3
aws s3 cp "$BACKUP_DIR/dump_$DATE.rdb" "$S3_BUCKET/dump_$DATE.rdb"

# 同步整个备份目录（可选）
aws s3 sync $BACKUP_DIR $S3_BUCKET --delete

# 设置 S3 生命周期（在 S3 控制台配置）
# - 30 天后转移到 Glacier
# - 90 天后删除
```

### 阿里云 OSS 备份

```bash
#!/bin/bash
# /opt/scripts/redis-backup-oss.sh

BACKUP_DIR="/backup/redis"
OSS_BUCKET="oss://your-bucket/redis-backup"
DATE=$(date +%Y%m%d_%H%M%S)

# 执行本地备份
/opt/scripts/redis-backup.sh

# 上传到 OSS
ossutil cp "$BACKUP_DIR/dump_$DATE.rdb" "$OSS_BUCKET/dump_$DATE.rdb"

# 或使用 ossutil sync 同步
ossutil sync $BACKUP_DIR $OSS_BUCKET
```

### 安装云存储工具

```bash
# AWS CLI
pip install awscli
aws configure

# 阿里云 ossutil
wget https://gosspublic.alicdn.com/ossutil/1.7.15/ossutil-v1.7.15-linux-amd64.zip
unzip ossutil-v1.7.15-linux-amd64.zip
sudo mv ossutil /usr/local/bin/
ossutil config
```

## 数据恢复

### RDB 恢复流程

```bash
# 1. 停止 Redis
sudo systemctl stop redis

# 2. 备份当前数据（可选）
cp /var/lib/redis/dump.rdb /var/lib/redis/dump.rdb.bak

# 3. 复制备份文件
cp /backup/redis/dump_20240101_020000.rdb /var/lib/redis/dump.rdb

# 4. 设置文件权限
chown redis:redis /var/lib/redis/dump.rdb
chmod 660 /var/lib/redis/dump.rdb

# 5. 启动 Redis
sudo systemctl start redis

# 6. 验证恢复
redis-cli INFO keyspace
redis-cli DBSIZE
```

### AOF 恢复流程

```bash
# 1. 停止 Redis
sudo systemctl stop redis

# 2. 备份当前数据
mv /var/lib/redis/appendonlydir /var/lib/redis/appendonlydir.bak

# 3. 恢复 AOF 文件 (Redis 7.0+)
tar -xzf /backup/redis/aof/aof_20240101_020000.tar.gz -C /var/lib/redis/

# 4. 设置权限
chown -R redis:redis /var/lib/redis/appendonlydir

# 5. 启动 Redis
sudo systemctl start redis

# 6. 验证恢复
redis-cli INFO persistence
```

### 修复损坏的持久化文件

**RDB 文件检查**：

```bash
# 检查 RDB 文件
redis-check-rdb dump.rdb

# 输出示例
# [offset 0] Checking RDB file dump.rdb
# [offset 26] AUX FIELD redis-ver = '7.0.0'
# ...
# [offset 1234567] Checksum OK
# [offset 1234567] \o/ RDB looks OK! \o/
```

**AOF 文件修复**：

```bash
# 检查并修复 AOF 文件
redis-check-aof --fix appendonly.aof

# Redis 7.0+ 修复
redis-check-aof --fix appendonlydir/appendonly.aof.manifest

# 修复后会生成备份
# appendonly.aof.bak (原文件备份)
```

### 从 RDB 恢复到 AOF 模式

```bash
# 1. 使用 RDB 启动 Redis
redis-server --dbfilename dump.rdb --appendonly no

# 2. 动态开启 AOF
redis-cli CONFIG SET appendonly yes

# 3. 等待 AOF 重写完成
redis-cli BGREWRITEAOF

# 4. 修改配置文件，永久生效
# appendonly yes
```

## 灾难恢复

### 恢复演练流程

定期（建议每季度）执行灾难恢复演练：

```bash
#!/bin/bash
# /opt/scripts/redis-dr-drill.sh
# 灾难恢复演练脚本

DR_ENV="dr-redis"
BACKUP_FILE="/backup/redis/dump_latest.rdb"
TEST_PORT=6380

echo "=== Redis 灾难恢复演练 ==="
echo "开始时间: $(date)"

# 1. 启动测试实例
echo "1. 启动测试 Redis 实例..."
docker run -d --name $DR_ENV \
    -p $TEST_PORT:6379 \
    -v $BACKUP_FILE:/data/dump.rdb:ro \
    redis:7.0 redis-server --dbfilename dump.rdb

sleep 5

# 2. 验证数据
echo "2. 验证数据..."
DBSIZE=$(redis-cli -p $TEST_PORT DBSIZE)
echo "   - 键数量: $DBSIZE"

# 3. 抽样检查
echo "3. 抽样检查..."
SAMPLE_KEYS=$(redis-cli -p $TEST_PORT SCAN 0 COUNT 5 | tail -n +2)
for key in $SAMPLE_KEYS; do
    TYPE=$(redis-cli -p $TEST_PORT TYPE "$key")
    echo "   - $key: $TYPE"
done

# 4. 性能测试
echo "4. 性能测试..."
redis-benchmark -p $TEST_PORT -t get,set -n 1000 -q

# 5. 清理
echo "5. 清理测试环境..."
docker stop $DR_ENV
docker rm $DR_ENV

echo "=== 演练完成 ==="
echo "结束时间: $(date)"
```

### 恢复时间目标 (RTO) 评估

| 恢复场景 | 预计 RTO | 影响因素 |
|----------|----------|----------|
| RDB 恢复（10GB） | 2-5 分钟 | 磁盘 I/O |
| AOF 恢复（10GB） | 5-15 分钟 | 命令重放速度 |
| 混合持久化恢复 | 2-5 分钟 | RDB 加载 + 增量 AOF |
| 从云存储恢复 | +5-30 分钟 | 网络下载时间 |

## 数据迁移

### 在线迁移（主从复制）

```bash
# 1. 新服务器配置为从节点
redis-cli -h new-server REPLICAOF old-server 6379

# 2. 等待同步完成
redis-cli -h new-server INFO replication
# master_link_status:up
# master_sync_in_progress:0

# 3. 提升为主节点
redis-cli -h new-server REPLICAOF NO ONE

# 4. 切换应用连接到新服务器
```

### 离线迁移（RDB 文件）

```bash
# 1. 在源服务器生成 RDB
redis-cli -h source-server BGSAVE

# 2. 复制 RDB 文件到目标服务器
scp source-server:/var/lib/redis/dump.rdb target-server:/var/lib/redis/

# 3. 在目标服务器启动 Redis
ssh target-server "systemctl start redis"
```

### 使用 redis-shake 迁移

```bash
# 下载 redis-shake
wget https://github.com/alibaba/RedisShake/releases/download/v3.1.0/redis-shake-linux-amd64.tar.gz
tar -xzf redis-shake-linux-amd64.tar.gz

# 配置 shake.toml
cat > shake.toml << EOF
[source]
address = "127.0.0.1:6379"
password = ""

[target]
address = "192.168.1.100:6379"
password = ""
EOF

# 执行迁移
./redis-shake shake.toml
```

## 集群环境备份

### Redis Cluster 备份

```bash
#!/bin/bash
# 集群备份脚本

CLUSTER_NODES="192.168.1.101:7001 192.168.1.102:7002 192.168.1.103:7003"
BACKUP_DIR="/backup/redis-cluster"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$DATE"

mkdir -p $BACKUP_PATH

for node in $CLUSTER_NODES; do
    HOST=$(echo $node | cut -d: -f1)
    PORT=$(echo $node | cut -d: -f2)
    
    echo "备份节点: $node"
    
    # 触发 BGSAVE
    redis-cli -h $HOST -p $PORT BGSAVE
    sleep 5
    
    # 复制 RDB 文件
    scp $HOST:/var/lib/redis/$PORT/dump.rdb "$BACKUP_PATH/dump_${HOST}_${PORT}.rdb"
done

# 保存集群信息
redis-cli -c -h 192.168.1.101 -p 7001 CLUSTER NODES > "$BACKUP_PATH/cluster_nodes.txt"
redis-cli -c -h 192.168.1.101 -p 7001 CLUSTER INFO > "$BACKUP_PATH/cluster_info.txt"

echo "集群备份完成: $BACKUP_PATH"
```

### Sentinel 环境备份

对于哨兵模式，只需备份主节点的 RDB/AOF 文件。从节点会自动从主节点同步数据。

```bash
# 获取当前主节点
MASTER=$(redis-cli -h sentinel-host -p 26379 SENTINEL get-master-addr-by-name mymaster)
MASTER_HOST=$(echo $MASTER | awk '{print $1}')
MASTER_PORT=$(echo $MASTER | awk '{print $2}')

# 备份主节点
redis-cli -h $MASTER_HOST -p $MASTER_PORT BGSAVE
```

## 监控与告警

### 备份监控指标

```bash
# 检查最后备份时间
redis-cli INFO persistence | grep rdb_last_save_time

# 检查备份文件
ls -la /backup/redis/

# 检查备份文件大小变化
du -sh /backup/redis/*
```

### 告警配置示例（Prometheus AlertManager）

```yaml
groups:
  - name: redis-backup
    rules:
      - alert: RedisBackupTooOld
        expr: time() - redis_rdb_last_save_timestamp_seconds > 86400
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Redis 备份超过 24 小时未更新"
          
      - alert: RedisBackupFileMissing
        expr: redis_rdb_last_cow_size_bytes == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Redis RDB 备份文件可能丢失"
```

## 小结

| 备份类型 | 适用场景 | 恢复速度 | 数据完整性 |
|----------|----------|----------|------------|
| RDB 手动 | 临时备份 | 快 | 中 |
| RDB 自动 | 定期备份 | 快 | 中 |
| AOF | 热备份 | 中 | 高 |
| 云存储 | 异地容灾 | 慢 | 高 |
| 主从复制 | 在线迁移 | 快 | 高 |

**最佳实践**：

- ✅ 同时启用 RDB 和 AOF
- ✅ 定期备份到异地存储
- ✅ 定期执行恢复演练
- ✅ 监控备份状态和文件大小
- ✅ 保留多个备份版本
