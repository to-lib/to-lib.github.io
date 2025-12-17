---
sidebar_position: 13
title: 监控与日志
description: Docker 容器监控、日志管理与可观测性
---

# Docker 监控与日志

本文介绍如何监控 Docker 容器的性能指标和管理容器日志。

## 内置监控工具

### docker stats

实时查看容器资源使用情况：

```bash
# 查看所有运行中容器
docker stats

# 查看特定容器
docker stats container1 container2

# 不刷新，只显示一次
docker stats --no-stream

# 自定义格式
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

输出指标：

| 指标              | 说明              |
| ----------------- | ----------------- |
| CPU %             | CPU 使用百分比    |
| MEM USAGE / LIMIT | 内存使用量 / 限制 |
| MEM %             | 内存使用百分比    |
| NET I/O           | 网络 I/O          |
| BLOCK I/O         | 磁盘 I/O          |
| PIDS              | 进程数            |

### docker top

查看容器内进程：

```bash
docker top container_name
docker top container_name -aux
```

### docker inspect

获取容器详细信息：

```bash
# 查看容器状态
docker inspect --format='{{.State.Status}}' container_name

# 查看 IP 地址
docker inspect --format='{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_name

# 查看资源限制
docker inspect --format='{{.HostConfig.Memory}}' container_name
```

## 日志管理

### 日志驱动类型

| 驱动      | 说明                        | 使用场景     |
| --------- | --------------------------- | ------------ |
| json-file | 默认，JSON 格式文件         | 开发测试     |
| syslog    | 输出到 syslog               | 系统日志集成 |
| journald  | 输出到 systemd journal      | systemd 系统 |
| fluentd   | 输出到 Fluentd              | 集中日志采集 |
| gelf      | Graylog Extended Log Format | Graylog 集成 |
| awslogs   | AWS CloudWatch Logs         | AWS 环境     |

### 配置日志驱动

**运行时配置**：

```bash
docker run -d \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  nginx
```

**全局配置** `/etc/docker/daemon.json`：

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3",
    "labels": "production_status",
    "env": "os,customer"
  }
}
```

### 查看日志

```bash
# 查看全部日志
docker logs container_name

# 实时跟踪
docker logs -f container_name

# 最近 N 行
docker logs --tail 100 container_name

# 指定时间范围
docker logs --since 2024-01-01T00:00:00 container_name
docker logs --since 30m container_name

# 显示时间戳
docker logs -t container_name
```

### Docker Compose 日志

```bash
# 所有服务日志
docker compose logs

# 特定服务
docker compose logs app db

# 实时跟踪
docker compose logs -f

# 限制行数
docker compose logs --tail 100
```

## Prometheus + Grafana 监控

### 部署架构

```
┌─────────────┐    ┌────────────────┐    ┌─────────────┐
│   Docker    │───▶│   cAdvisor     │───▶│  Prometheus │
│  Containers │    │                │    │             │
└─────────────┘    └────────────────┘    └──────┬──────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │   Grafana   │
                                         └─────────────┘
```

### 部署配置

```yaml
# docker-compose.monitoring.yml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
    ports:
      - "9090:9090"
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - monitoring

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    ports:
      - "8080:8080"
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:latest
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - "--path.procfs=/host/proc"
      - "--path.sysfs=/host/sys"
    ports:
      - "9100:9100"
    networks:
      - monitoring

volumes:
  prometheus-data:
  grafana-data:

networks:
  monitoring:
```

### Prometheus 配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "cadvisor"
    static_configs:
      - targets: ["cadvisor:8080"]

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]
```

### 常用监控指标

**容器 CPU**：

```promql
# CPU 使用率
rate(container_cpu_usage_seconds_total{name!=""}[5m]) * 100

# 按容器分组
sum(rate(container_cpu_usage_seconds_total{name!=""}[5m])) by (name) * 100
```

**容器内存**：

```promql
# 内存使用量
container_memory_usage_bytes{name!=""}

# 内存使用率
container_memory_usage_bytes{name!=""} / container_spec_memory_limit_bytes{name!=""} * 100
```

**网络 I/O**：

```promql
# 接收流量
rate(container_network_receive_bytes_total{name!=""}[5m])

# 发送流量
rate(container_network_transmit_bytes_total{name!=""}[5m])
```

## 集中日志采集

### ELK Stack

```yaml
# docker-compose.elk.yml
version: "3.8"

services:
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - es-data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  logstash:
    image: logstash:8.11.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  es-data:
```

### Fluentd 配置

```yaml
# 使用 fluentd 日志驱动
services:
  app:
    image: myapp
    logging:
      driver: fluentd
      options:
        fluentd-address: "localhost:24224"
        tag: "docker.{{.Name}}"
```

### Loki + Promtail

```yaml
# docker-compose.loki.yml
version: "3.8"

services:
  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"
    volumes:
      - loki-data:/loki

  promtail:
    image: grafana/promtail:2.9.0
    volumes:
      - /var/log:/var/log
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - ./promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml

volumes:
  loki-data:
```

## 告警配置

### Prometheus Alertmanager

```yaml
# alertmanager.yml
global:
  smtp_smarthost: "smtp.example.com:587"
  smtp_from: "alertmanager@example.com"

route:
  receiver: "email-notifications"

receivers:
  - name: "email-notifications"
    email_configs:
      - to: "admin@example.com"
```

### 告警规则

```yaml
# alert-rules.yml
groups:
  - name: container-alerts
    rules:
      - alert: ContainerHighCPU
        expr: rate(container_cpu_usage_seconds_total{name!=""}[5m]) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Container CPU usage high"
          description: "Container {{ $labels.name }} CPU usage is above 80%"

      - alert: ContainerHighMemory
        expr: container_memory_usage_bytes{name!=""} / container_spec_memory_limit_bytes{name!=""} * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Container memory usage high"
          description: "Container {{ $labels.name }} memory usage is above 80%"
```

## 健康检查

### Dockerfile 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

### Docker Compose 健康检查

```yaml
services:
  app:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 查看健康状态

```bash
# 查看健康状态
docker inspect --format='{{.State.Health.Status}}' container_name

# 查看健康检查日志
docker inspect --format='{{json .State.Health}}' container_name | jq
```

## 监控最佳实践

1. **设置资源限制**：便于监控和告警
2. **使用健康检查**：自动检测服务状态
3. **集中日志管理**：便于问题排查
4. **设置合理告警阈值**：避免告警疲劳
5. **定期审查监控指标**：优化资源分配
