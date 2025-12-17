---
sidebar_position: 24
title: 配置与部署
---

# Redis 配置与部署

本章聚焦 Redis 在生产环境最常用、最容易踩坑的配置项与部署要点，帮助你把 Redis 从“能跑”配置到“可用、可控、可维护”。

## 配置文件位置

- Linux 发行版安装：通常为 `/etc/redis/redis.conf`
- Homebrew（macOS）：通常为 `/usr/local/etc/redis.conf` 或 `/opt/homebrew/etc/redis.conf`
- Docker：常用做法是把 `redis.conf` 挂载进容器

## 启动方式与加载配置

```bash
# 指定配置文件启动
redis-server /etc/redis/redis.conf

# 运行中查看关键配置（部分配置可能被动态修改）
redis-cli CONFIG GET '*maxmemory*'
```

## 网络与连接

### bind / protected-mode

生产环境建议：

- Redis 只监听内网 IP
- 保持 `protected-mode yes`
- 不要直接暴露到公网

```conf
bind 127.0.0.1 192.168.1.10
protected-mode yes
port 6379
```

### timeout / tcp-backlog

```conf
# 连接空闲超时（秒），0 表示不超时
timeout 300

# TCP 半连接队列
tcp-backlog 511
```

## 认证与权限

### requirepass

```conf
requirepass your_strong_password_here
```

更推荐使用 ACL（Redis 6+）：

```bash
ACL SETUSER app on >app_password ~app:* +@all -@dangerous
```

更多安全细节见：[安全配置](/docs/redis/security)

## 内存与淘汰策略

### maxmemory / maxmemory-policy

```conf
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5
```

- 如果 Redis 主要用于缓存：通常选择 `allkeys-lru` 或 `allkeys-lfu`
- 如果数据不能丢：使用 `noeviction`，并在写入报错时由业务兜底

更多内存与大 Key 治理见：[内存管理](/docs/redis/memory-management)

## 持久化

### RDB（快照）

```conf
save 900 1
save 300 10
save 60 10000
```

### AOF（追加日志）

```conf
appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite yes
```

更多策略对比见：[持久化](/docs/redis/persistence)

## 复制与高可用（关键参数）

### 主从复制

```conf
replicaof <master-ip> 6379
masterauth <master-password>
replica-read-only yes
repl-timeout 60
```

更多见：[主从复制](/docs/redis/replication)

### Sentinel / Cluster

- Sentinel：见 [哨兵模式](/docs/redis/sentinel)
- Cluster：见 [Redis 集群](/docs/redis/cluster)

## 日志与诊断

```conf
loglevel notice
logfile /var/log/redis/redis.log
```

建议把排障流程固化到手册：

- [监控与排障](/docs/redis/monitoring-and-troubleshooting)

## 推荐的最小生产配置（示例）

```conf
bind 127.0.0.1 192.168.1.10
protected-mode yes
port 6379

requirepass your_strong_password_here

maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite yes

slowlog-log-slower-than 10000
slowlog-max-len 128
```
