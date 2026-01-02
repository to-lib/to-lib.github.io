---
sidebar_position: 24
title: 生产实践
description: 容器编排选型、蓝绿部署与灾备迁移
---

# 生产实践

Docker 在生产环境中的最佳实践和部署策略。

## 编排工具选型

### Docker Swarm vs Kubernetes

| 特性 | Docker Swarm | Kubernetes |
|------|-------------|------------|
| 学习曲线 | 低 | 高 |
| 安装复杂度 | 简单 | 复杂 |
| 功能丰富度 | 基础 | 全面 |
| 社区生态 | 较小 | 庞大 |
| 适用规模 | 中小型 | 任意规模 |
| 高可用 | 支持 | 支持 |
| 服务发现 | 内置 | 内置 |
| 负载均衡 | 内置 | 内置 + Ingress |

### 选型建议

**选择 Docker Swarm：**
- 团队规模小，运维能力有限
- 应用规模中等（几十个服务）
- 需要快速上手
- 已有 Docker Compose 配置

**选择 Kubernetes：**
- 大规模微服务架构
- 需要高级调度策略
- 多云/混合云部署
- 需要丰富的生态工具

## 蓝绿部署

### 使用 Docker Compose

```yaml
# docker-compose.blue.yml
services:
  app:
    image: myapp:v1
    deploy:
      replicas: 3
    networks:
      - app-network
    labels:
      - "traefik.http.routers.app.rule=Host(`app.example.com`)"
```

```yaml
# docker-compose.green.yml
services:
  app:
    image: myapp:v2
    deploy:
      replicas: 3
    networks:
      - app-network
    labels:
      - "traefik.http.routers.app-green.rule=Host(`app.example.com`)"
```

部署流程：

```bash
# 1. 部署绿色环境
docker compose -f docker-compose.green.yml up -d

# 2. 测试绿色环境
curl http://green.app.example.com/health

# 3. 切换流量（更新 Traefik 配置或 DNS）

# 4. 确认无误后，停止蓝色环境
docker compose -f docker-compose.blue.yml down
```

### 使用 Docker Swarm

```bash
# 创建服务
docker service create \
  --name app \
  --replicas 3 \
  --update-delay 10s \
  --update-parallelism 1 \
  myapp:v1

# 蓝绿部署：创建新服务
docker service create \
  --name app-green \
  --replicas 3 \
  myapp:v2

# 测试后切换（更新负载均衡器指向）

# 删除旧服务
docker service rm app
docker service update --name app app-green
```

## 金丝雀部署

### Docker Swarm 实现

```bash
# 初始部署
docker service create \
  --name app \
  --replicas 10 \
  myapp:v1

# 金丝雀：更新部分实例
docker service update \
  --replicas 10 \
  --update-parallelism 1 \
  --image myapp:v2 \
  app

# 观察指标，确认无误后继续
docker service update \
  --update-parallelism 3 \
  --image myapp:v2 \
  app
```

### 使用 Traefik 权重路由

```yaml
services:
  app-v1:
    image: myapp:v1
    labels:
      - "traefik.http.services.app-v1.loadbalancer.server.weight=90"

  app-v2:
    image: myapp:v2
    labels:
      - "traefik.http.services.app-v2.loadbalancer.server.weight=10"
```

## 滚动更新

### Docker Swarm 配置

```bash
docker service update \
  --update-delay 10s \
  --update-parallelism 2 \
  --update-failure-action rollback \
  --update-max-failure-ratio 0.2 \
  --image myapp:v2 \
  app
```

### Docker Compose (Swarm mode)

```yaml
services:
  app:
    image: myapp:v2
    deploy:
      replicas: 6
      update_config:
        parallelism: 2
        delay: 10s
        failure_action: rollback
        max_failure_ratio: 0.2
        order: start-first
      rollback_config:
        parallelism: 1
        delay: 5s
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
    image: myapp
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
```

## 灾备与迁移

### 数据备份策略

```bash
# 定时备份脚本
#!/bin/bash
BACKUP_DIR=/backups/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# 备份数据卷
for volume in $(docker volume ls -q); do
  docker run --rm \
    -v $volume:/data \
    -v $BACKUP_DIR:/backup \
    alpine tar czf /backup/$volume.tar.gz -C /data .
done

# 备份容器配置
docker inspect $(docker ps -aq) > $BACKUP_DIR/containers.json

# 清理旧备份（保留 7 天）
find /backups -type d -mtime +7 -exec rm -rf {} +
```

### 数据恢复

```bash
# 恢复数据卷
docker volume create mydata
docker run --rm \
  -v mydata:/data \
  -v /backups/20240101:/backup \
  alpine tar xzf /backup/mydata.tar.gz -C /data
```

### 跨主机迁移

```bash
# 源主机：导出
docker save myapp:latest | gzip > myapp.tar.gz
docker run --rm -v mydata:/data -v /tmp:/backup alpine \
  tar czf /backup/mydata.tar.gz -C /data .

# 传输
scp myapp.tar.gz mydata.tar.gz user@newhost:/tmp/

# 目标主机：导入
docker load < /tmp/myapp.tar.gz
docker volume create mydata
docker run --rm -v mydata:/data -v /tmp:/backup alpine \
  tar xzf /backup/mydata.tar.gz -C /data
```

## 高可用配置

### Docker Swarm 高可用

```bash
# 初始化 Swarm（第一个 manager）
docker swarm init --advertise-addr 192.168.1.1

# 添加 manager 节点（至少 3 个实现高可用）
docker swarm join-token manager
# 在其他节点执行输出的命令

# 添加 worker 节点
docker swarm join-token worker

# 查看节点状态
docker node ls
```

### 服务高可用

```yaml
services:
  app:
    image: myapp
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.role == worker
        preferences:
          - spread: node.labels.zone
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
```

## 监控告警

### 基础监控栈

```yaml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  alertmanager:
    image: prom/alertmanager
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    ports:
      - "9093:9093"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  node-exporter:
    image: prom/node-exporter
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
```

### 告警规则示例

```yaml
# prometheus/alerts.yml
groups:
  - name: container-alerts
    rules:
      - alert: ContainerHighCPU
        expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Container CPU usage high"

      - alert: ContainerHighMemory
        expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Container memory usage high"

      - alert: ContainerDown
        expr: absent(container_last_seen{name=~".+"})
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Container is down"
```
