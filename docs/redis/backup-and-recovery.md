---
sidebar_position: 26
title: 备份与恢复
---

# Redis 备份与恢复

Redis 的备份/恢复通常围绕两类持久化文件：

- RDB（快照）：`dump.rdb`
- AOF（追加日志）：`appendonly.aof`（以及 Redis 7 的多文件 AOF 目录结构）

持久化机制与参数详见：[持久化](/docs/redis/persistence)

## 备份策略选择

- **缓存场景**：可只开 RDB（甚至关闭持久化），重点是快速扩容/重建
- **重要数据**：建议同时开启 AOF（everysec）+ RDB，并定期做离线备份

## RDB 备份

### 手动触发快照

```bash
redis-cli BGSAVE
```

### 备份文件

1. 确认 RDB 文件路径（配置项 `dir`、`dbfilename`）
2. 将 `dump.rdb` 复制到备份介质（对象存储/NAS/备份服务器）

## AOF 备份

### 手动触发重写

```bash
redis-cli BGREWRITEAOF
```

### 备份建议

- 备份前先执行 AOF rewrite，降低文件体积
- 备份时避免高峰期（rewrite 会带来额外 IO/CPU）

## 恢复流程（通用）

### 1. 停止 Redis

根据你的部署方式停止服务（systemd/Docker/k8s）。

### 2. 放置持久化文件

- 把备份的 `dump.rdb`/AOF 文件放回 Redis 的 `dir` 目录
- 注意文件权限与属主（例如 `redis:redis`）

### 3. 启动 Redis 并验证

```bash
redis-cli PING
redis-cli INFO persistence
```

## 灾难恢复建议（生产）

- 定期演练：备份文件是否能在新机器上恢复
- 多副本存储：至少 2 份异地备份
- 备份监控：备份任务成功率、文件大小变化、恢复耗时

## 数据迁移（简要）

常见迁移方式：

- **RDB 文件迁移**：适合冷迁移/停机窗口
- **主从复制迁移**：通过临时从库同步数据，切换连接
- **集群 reshard**：在线扩缩容

迁移过程容易引入一致性风险，建议结合：[监控与排障](/docs/redis/monitoring-and-troubleshooting)
