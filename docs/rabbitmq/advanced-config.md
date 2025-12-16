---
sidebar_position: 11
title: "高级配置"
description: "RabbitMQ 服务端配置、策略、资源限制与常用参数"
---

# 高级配置

本文聚焦 RabbitMQ 服务端侧的常用高级配置：`rabbitmq.conf`、策略（policy）、资源限制、以及生产环境建议。

## 配置文件位置

常见路径（不同安装方式会略有差异）：

- Linux：`/etc/rabbitmq/rabbitmq.conf`
- Docker：通常通过挂载或环境变量注入

## rabbitmq.conf 常用配置片段

```ini
# 连接与心跳
heartbeat = 60
channel_max = 2047

# 内存与磁盘阈值
vm_memory_high_watermark.relative = 0.6
vm_memory_high_watermark_paging_ratio = 0.5

disk_free_limit.absolute = 5GB

# 网络
listeners.tcp.default = 5672
management.tcp.port = 15672

# 集群网络分区处理
cluster_partition_handling = autoheal

# 日志
log.file.level = info
```

## Policy（策略）

策略可以对匹配的队列/交换机批量应用参数。

```bash
# 查看策略
rabbitmqctl list_policies

# 设置策略（示例：对 app. 前缀队列设置 TTL 与 DLX）
rabbitmqctl set_policy app-ttl "^app\\." '{"message-ttl":60000,"dead-letter-exchange":"dlx.exchange"}' --apply-to queues

# 清除策略
rabbitmqctl clear_policy app-ttl
```

:::warning 注意
策略的键名与 `arguments` 不完全一致。例如：policy 中常用 `message-ttl`，而声明队列 arguments 常用 `x-message-ttl`。
:::

## Resource Limits（资源限制）

限制连接数/队列数等，防止某个租户或误操作拖垮集群。

```bash
# 用户限制
rabbitmqctl set_user_limits app_user '{"max-connections":200}'

# vhost 限制
rabbitmqctl set_vhost_limits /app '{"max-connections":1000,"max-queues":5000}'

rabbitmqctl list_user_limits
rabbitmqctl list_vhost_limits
```

## Definitions（导入/导出）

用于备份/迁移（交换机、队列、绑定、用户、权限等）。

```bash
rabbitmqctl export_definitions definitions.json
rabbitmqctl import_definitions definitions.json
```

也可通过 Management HTTP API 进行备份与恢复（见 `monitoring`）。

## TLS（生产环境强烈建议）

TLS 相关细节与证书生成请参考：

- 🔐 [安全配置](/docs/rabbitmq/security)

## 下一步

- 🔧 [高级特性](/docs/rabbitmq/advanced-features) - 延迟、重试、DLX、幂等等组合用法
- 📊 [监控运维](/docs/rabbitmq/monitoring) - 指标、告警与备份
- ⚙️ [集群管理](/docs/rabbitmq/cluster-management) - 高可用与扩容
