---
sidebar_position: 13
title: 高可用架构
---

# PostgreSQL 高可用架构

高可用（High Availability）确保数据库服务在故障时能够快速恢复，最大限度减少停机时间。

## 📚 高可用方案对比

| 方案                 | 自动故障转移 | 复杂度 | 适用场景     |
| -------------------- | ------------ | ------ | ------------ |
| **Patroni + etcd**   | ✅           | 中     | 生产环境首选 |
| **Repmgr**           | ✅           | 中     | 传统方案     |
| **pg_auto_failover** | ✅           | 低     | 简单场景     |
| **PgPool-II**        | ✅           | 高     | 需要连接池   |
| **Cloud Managed**    | ✅           | 低     | 云环境       |

## 🏆 Patroni + etcd（推荐方案）

Patroni 是目前最流行的 PostgreSQL 高可用方案，由 Zalando 开发。

### 架构图

```
                    ┌─────────────┐
                    │   HAProxy   │
                    │ (负载均衡)   │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
    │ Patroni │       │ Patroni │       │ Patroni │
    │  Node1  │       │  Node2  │       │  Node3  │
    │ (Leader)│       │(Replica)│       │(Replica)│
    └────┬────┘       └────┬────┘       └────┬────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                    ┌──────▼──────┐
                    │    etcd     │
                    │  (分布式KV) │
                    └─────────────┘
```

### 1. 安装 etcd

```bash
# Ubuntu/Debian
sudo apt-get install etcd

# 配置 etcd 集群
cat > /etc/etcd/etcd.conf.yml << EOF
name: etcd1
data-dir: /var/lib/etcd
listen-peer-urls: http://10.0.0.1:2380
listen-client-urls: http://10.0.0.1:2379,http://127.0.0.1:2379
advertise-client-urls: http://10.0.0.1:2379
initial-advertise-peer-urls: http://10.0.0.1:2380
initial-cluster: etcd1=http://10.0.0.1:2380,etcd2=http://10.0.0.2:2380,etcd3=http://10.0.0.3:2380
initial-cluster-token: etcd-cluster-1
initial-cluster-state: new
EOF

sudo systemctl start etcd
```

### 2. 安装 Patroni

```bash
# 安装
pip3 install patroni[etcd]

# 或使用系统包
sudo apt-get install patroni
```

### 3. 配置 Patroni

**patroni.yml：**

```yaml
scope: postgres-cluster
name: node1

restapi:
  listen: 0.0.0.0:8008
  connect_address: 10.0.0.1:8008

etcd:
  hosts:
    - 10.0.0.1:2379
    - 10.0.0.2:2379
    - 10.0.0.3:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 1048576
    postgresql:
      use_pg_rewind: true
      use_slots: true
      parameters:
        wal_level: replica
        hot_standby: on
        max_connections: 200
        max_wal_senders: 10
        max_replication_slots: 10
        wal_keep_size: 1024

  initdb:
    - encoding: UTF8
    - data-checksums

  pg_hba:
    - host replication replicator 10.0.0.0/24 md5
    - host all all 10.0.0.0/24 md5

  users:
    admin:
      password: admin_password
      options:
        - createrole
        - createdb
    replicator:
      password: replicator_password
      options:
        - replication

postgresql:
  listen: 0.0.0.0:5432
  connect_address: 10.0.0.1:5432
  data_dir: /var/lib/postgresql/16/main
  bin_dir: /usr/lib/postgresql/16/bin

  authentication:
    replication:
      username: replicator
      password: replicator_password
    superuser:
      username: postgres
      password: postgres_password
    rewind:
      username: postgres
      password: postgres_password

  parameters:
    unix_socket_directories: "/var/run/postgresql"

tags:
  nofailover: false
  noloadbalance: false
  clonefrom: false
  nosync: false
```

### 4. 启动 Patroni

```bash
# 使用 systemd
sudo systemctl start patroni

# 查看集群状态
patronictl -c /etc/patroni/patroni.yml list
```

### 5. Patroni 常用命令

```bash
# 查看集群状态
patronictl -c patroni.yml list

# 手动切换主库
patronictl -c patroni.yml switchover

# 强制故障转移
patronictl -c patroni.yml failover

# 重新初始化节点
patronictl -c patroni.yml reinit node2

# 暂停自动故障转移
patronictl -c patroni.yml pause

# 恢复自动故障转移
patronictl -c patroni.yml resume
```

## 🔄 HAProxy 配置

HAProxy 提供负载均衡和自动路由。

### haproxy.cfg

```conf
global
    maxconn 1000

defaults
    mode tcp
    timeout connect 10s
    timeout client 30s
    timeout server 30s

listen stats
    bind *:7000
    mode http
    stats enable
    stats uri /

# 主库（写操作）
listen postgres-primary
    bind *:5000
    option httpchk GET /master
    http-check expect status 200
    default-server inter 3s fall 3 rise 2
    server node1 10.0.0.1:5432 check port 8008
    server node2 10.0.0.2:5432 check port 8008
    server node3 10.0.0.3:5432 check port 8008

# 从库（读操作）
listen postgres-replica
    bind *:5001
    balance roundrobin
    option httpchk GET /replica
    http-check expect status 200
    default-server inter 3s fall 3 rise 2
    server node1 10.0.0.1:5432 check port 8008
    server node2 10.0.0.2:5432 check port 8008
    server node3 10.0.0.3:5432 check port 8008
```

### 应用连接配置

```
# 写操作连接主库
postgresql://user:pass@haproxy:5000/dbname

# 读操作连接从库
postgresql://user:pass@haproxy:5001/dbname
```

## 🔌 PgBouncer 连接池

PgBouncer 减少数据库连接开销。

### pgbouncer.ini

```ini
[databases]
mydb = host=127.0.0.1 port=5432 dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
min_pool_size = 5
reserve_pool_size = 5
```

### userlist.txt

```
"postgres" "md5password_hash"
"myuser" "md5password_hash"
```

生成密码哈希：

```bash
echo -n "passwordusername" | md5sum | awk '{print "md5" $1}'
```

## 🛡️ pg_auto_failover

Citus 提供的简单高可用方案。

### 1. 安装

```bash
# 添加仓库
curl https://install.citusdata.com/community/deb.sh | sudo bash

# 安装
sudo apt-get install postgresql-16-auto-failover
```

### 2. 创建 Monitor 节点

```bash
pg_autoctl create monitor \
    --pgdata /var/lib/postgresql/monitor \
    --hostname monitor.example.com
```

### 3. 创建数据节点

```bash
# 主节点
pg_autoctl create postgres \
    --pgdata /var/lib/postgresql/data \
    --hostname node1.example.com \
    --monitor postgres://autoctl_node@monitor.example.com/pg_auto_failover

# 从节点
pg_autoctl create postgres \
    --pgdata /var/lib/postgresql/data \
    --hostname node2.example.com \
    --monitor postgres://autoctl_node@monitor.example.com/pg_auto_failover
```

### 4. 查看状态

```bash
pg_autoctl show state
pg_autoctl show events
```

## ☁️ 云服务高可用

### AWS RDS

- **Multi-AZ 部署**：自动故障转移
- **Read Replicas**：读写分离
- **自动备份**：时间点恢复

### Google Cloud SQL

- **高可用配置**：自动故障转移
- **读取副本**：跨区域复制
- **自动备份**：7 天保留

### Azure Database

- **区域冗余**：高可用
- **只读副本**：最多 5 个
- **自动备份**：35 天保留

## 📊 监控高可用

### 关键指标

```sql
-- 复制延迟
SELECT client_addr,
       pg_wal_lsn_diff(sent_lsn, replay_lsn) AS lag_bytes,
       EXTRACT(EPOCH FROM (now() - replay_lag)) AS lag_seconds
FROM pg_stat_replication;

-- 连接数
SELECT count(*) FROM pg_stat_activity;

-- 长事务
SELECT pid, now() - xact_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active'
  AND xact_start < now() - interval '5 minutes';
```

### 告警配置

- 复制延迟 > 30 秒
- 连接数 > 80%
- 主库不可用
- 从库数量减少

## 💡 最佳实践

1. **至少 3 个节点**：避免脑裂
2. **跨可用区部署**：提高容灾能力
3. **定期故障转移演练**：确保流程有效
4. **监控复制延迟**：及时发现问题
5. **使用连接池**：减少连接开销

## 📚 相关资源

- [主从复制](/docs/postgres/replication) - 复制配置详解
- [备份恢复](/docs/postgres/backup-recovery) - 数据备份策略
- [性能优化](/docs/postgres/performance-optimization) - 性能调优
