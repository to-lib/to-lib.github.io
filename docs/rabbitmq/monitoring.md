---
sidebar_position: 10
title: "监控运维"
description: "RabbitMQ 监控与运维指南"
---

# RabbitMQ 监控运维

本指南介绍 RabbitMQ 的监控和运维管理。

## 管理界面

### 启用管理插件

```bash
rabbitmq-plugins enable rabbitmq_management
```

访问地址: `http://localhost:15672`

### 主要功能

- 概览：节点状态、消息速率
- 连接：查看和管理连接
- 通道：查看通道信息
- 交换机：管理交换机
- 队列：队列状态和管理
- 管理：用户、策略、集群

## 命令行监控

### 节点状态

```bash
# 查看节点状态
rabbitmqctl status

# 查看集群状态
rabbitmqctl cluster_status

# 健康检查
rabbitmqctl node_health_check
```

### 队列信息

```bash
# 列出所有队列
rabbitmqctl list_queues name messages consumers memory

# 查看队列详情
rabbitmqctl list_queues name \
  messages_ready \
  messages_unacknowledged \
  consumers \
  memory
```

### 连接信息

```bash
# 列出连接
rabbitmqctl list_connections user peer_host state

# 列出通道
rabbitmqctl list_channels connection consumer_count messages_unacknowledged
```

## Prometheus 监控

### 启用插件

```bash
rabbitmq-plugins enable rabbitmq_prometheus
```

### 指标端点

- `http://localhost:15692/metrics` - Prometheus 格式

### Grafana 仪表盘

导入官方仪表盘 ID: `10991`

### 关键指标

| 指标                                  | 说明       | 告警阈值 |
| ------------------------------------- | ---------- | -------- |
| `rabbitmq_queue_messages`             | 队列消息数 | > 10000  |
| `rabbitmq_connections`                | 连接数     | > 1000   |
| `rabbitmq_channels`                   | 通道数     | > 5000   |
| `rabbitmq_consumers`                  | 消费者数   | = 0      |
| `rabbitmq_node_mem_used`              | 内存使用   | > 80%    |
| `rabbitmq_disk_space_available_bytes` | 磁盘空间   | < 5GB    |

## 告警配置

### Prometheus 告警规则

```yaml
groups:
  - name: rabbitmq
    rules:
      - alert: RabbitMQDown
        expr: up{job="rabbitmq"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "RabbitMQ 服务不可用"

      - alert: RabbitMQQueueBacklog
        expr: rabbitmq_queue_messages > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "队列消息积压: {{ $value }}"

      - alert: RabbitMQNoConsumers
        expr: rabbitmq_queue_consumers == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "队列没有消费者"

      - alert: RabbitMQHighMemory
        expr: rabbitmq_node_mem_used / rabbitmq_node_mem_limit > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RabbitMQ 内存使用率过高"
```

## 日志管理

### 日志位置

```bash
# Linux
/var/log/rabbitmq/

# Docker
docker logs rabbitmq
```

### 日志配置

```ini
# rabbitmq.conf
log.file.level = info
log.console = true
log.console.level = warning

# 日志轮转
log.file.rotation.date = $D0
log.file.rotation.size = 10485760
log.file.rotation.count = 10
```

## 备份恢复

### 导出定义

```bash
# 导出所有定义（交换机、队列、绑定、用户等）
rabbitmqctl export_definitions /path/to/definitions.json

# 或通过 HTTP API
curl -u admin:password \
  http://localhost:15672/api/definitions \
  > definitions.json
```

### 导入定义

```bash
# 导入定义
rabbitmqctl import_definitions /path/to/definitions.json

# 或通过 HTTP API
curl -u admin:password \
  -X POST \
  -H "Content-Type: application/json" \
  -d @definitions.json \
  http://localhost:15672/api/definitions
```

### 数据备份

```bash
# 停止服务
rabbitmqctl stop_app

# 备份数据目录
tar -czvf rabbitmq-backup.tar.gz /var/lib/rabbitmq/mnesia/

# 启动服务
rabbitmqctl start_app
```

## 运维脚本

### 健康检查脚本

```bash
#!/bin/bash
# health_check.sh

# 检查服务状态
if ! rabbitmqctl status > /dev/null 2>&1; then
    echo "RabbitMQ is DOWN"
    exit 1
fi

# 检查队列深度
QUEUE_DEPTH=$(rabbitmqctl list_queues name messages --formatter json | \
    jq '[.[].messages] | add')

if [ "$QUEUE_DEPTH" -gt 100000 ]; then
    echo "Warning: High queue depth: $QUEUE_DEPTH"
    exit 2
fi

echo "RabbitMQ is healthy"
exit 0
```

### 清理脚本

```bash
#!/bin/bash
# cleanup.sh

# 清理未使用的队列
rabbitmqctl list_queues name consumers --quiet | \
    awk '$2 == 0 {print $1}' | \
    while read queue; do
        echo "Deleting unused queue: $queue"
        rabbitmqctl delete_queue "$queue"
    done
```

## 故障排查

### 常见问题

```bash
# 检查文件句柄
rabbitmqctl eval 'file:get_cwd().'
ulimit -n

# 检查网络连接
netstat -an | grep 5672

# 检查 Erlang 进程
ps aux | grep beam
```

### 日志分析

```bash
# 查看错误日志
grep -i error /var/log/rabbitmq/*.log

# 查看连接问题
grep -i "connection" /var/log/rabbitmq/*.log
```

## 下一步

- ❓ [常见问题](/docs/rabbitmq/faq) - FAQ
- 💼 [面试题集](/docs/rabbitmq/interview-questions) - 面试常见问题

## 参考资料

- [RabbitMQ 监控](https://www.rabbitmq.com/monitoring.html)
- [Prometheus 插件](https://www.rabbitmq.com/prometheus.html)
