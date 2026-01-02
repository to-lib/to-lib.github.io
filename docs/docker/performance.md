---
sidebar_position: 22
title: 性能调优
description: 容器性能分析、日志管理与资源配额详解
---

# 性能调优

掌握 Docker 容器的性能监控、分析和优化技巧。

## 性能监控工具

### docker stats

```bash
# 实时查看所有容器资源使用
docker stats

# 查看特定容器
docker stats container1 container2

# 不刷新，只输出一次
docker stats --no-stream

# 自定义输出格式
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

输出字段说明：

| 字段 | 说明 |
|------|------|
| CPU % | CPU 使用百分比 |
| MEM USAGE / LIMIT | 内存使用量 / 限制 |
| MEM % | 内存使用百分比 |
| NET I/O | 网络 I/O |
| BLOCK I/O | 磁盘 I/O |
| PIDS | 进程数 |

### cAdvisor

Google 开源的容器监控工具。

```bash
# 运行 cAdvisor
docker run -d \
  --name cadvisor \
  --privileged \
  -p 8080:8080 \
  -v /:/rootfs:ro \
  -v /var/run:/var/run:ro \
  -v /sys:/sys:ro \
  -v /var/lib/docker/:/var/lib/docker:ro \
  gcr.io/cadvisor/cadvisor:latest

# 访问 http://localhost:8080
```

### Prometheus + Grafana

```yaml
# docker-compose.yml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana

  cadvisor:
    image: gcr.io/cadvisor/cadvisor
    privileged: true
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro

volumes:
  prometheus-data:
  grafana-data:
```

## 日志管理

### 日志驱动类型

| 驱动 | 说明 | 适用场景 |
|------|------|----------|
| `json-file` | 默认，JSON 格式文件 | 开发环境 |
| `local` | 优化的本地日志 | 生产环境本地存储 |
| `syslog` | 发送到 syslog | 集中式日志 |
| `journald` | 发送到 systemd journal | systemd 系统 |
| `fluentd` | 发送到 Fluentd | EFK 栈 |
| `awslogs` | 发送到 CloudWatch | AWS 环境 |
| `gcplogs` | 发送到 Google Cloud | GCP 环境 |

### 配置日志驱动

```json
// /etc/docker/daemon.json - 全局配置
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3",
    "compress": "true"
  }
}
```

```bash
# 运行时指定
docker run -d \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  nginx
```

### 日志轮转配置

```json
{
  "log-driver": "local",
  "log-opts": {
    "max-size": "50m",
    "max-file": "5"
  }
}
```

### Docker Compose 日志配置

```yaml
services:
  app:
    image: myapp
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

### 集中式日志 (Fluentd)

```yaml
services:
  app:
    image: myapp
    logging:
      driver: fluentd
      options:
        fluentd-address: localhost:24224
        tag: docker.{{.Name}}

  fluentd:
    image: fluent/fluentd
    ports:
      - "24224:24224"
    volumes:
      - ./fluent.conf:/fluentd/etc/fluent.conf
```

## 资源限制

### CPU 限制

```bash
# 限制 CPU 核心数
docker run --cpus=2 nginx  # 最多使用 2 个 CPU

# CPU 份额（相对权重）
docker run --cpu-shares=512 nginx  # 默认 1024

# 绑定到特定 CPU
docker run --cpuset-cpus="0,1" nginx  # 只使用 CPU 0 和 1

# CPU 周期限制
docker run --cpu-period=100000 --cpu-quota=50000 nginx  # 50% CPU
```

### 内存限制

```bash
# 硬限制
docker run --memory=512m nginx

# 内存 + Swap 限制
docker run --memory=512m --memory-swap=1g nginx

# 禁用 Swap
docker run --memory=512m --memory-swap=512m nginx

# 内存软限制（OOM 优先级）
docker run --memory=512m --memory-reservation=256m nginx

# OOM 优先级调整
docker run --oom-score-adj=-500 nginx  # 降低被 OOM kill 的概率
```

### I/O 限制

```bash
# 限制读写速率
docker run \
  --device-read-bps /dev/sda:10mb \
  --device-write-bps /dev/sda:10mb \
  nginx

# 限制 IOPS
docker run \
  --device-read-iops /dev/sda:1000 \
  --device-write-iops /dev/sda:1000 \
  nginx

# 块 I/O 权重
docker run --blkio-weight=500 nginx  # 默认 500，范围 10-1000
```

### 进程数限制

```bash
# 限制容器内进程数（防止 fork 炸弹）
docker run --pids-limit=100 nginx
```

### Docker Compose 资源限制

```yaml
services:
  app:
    image: myapp
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 512M
          pids: 100
        reservations:
          cpus: "0.5"
          memory: 256M
```

## 性能分析

### 容器内性能分析

```bash
# 安装性能工具
docker exec container apt-get update && apt-get install -y \
  htop \
  iotop \
  strace \
  perf

# 查看进程
docker exec container htop

# 查看 I/O
docker exec container iotop

# 系统调用追踪
docker exec container strace -p 1
```

### 使用 perf 分析

```bash
# 主机上运行 perf（需要 --privileged 或 --cap-add SYS_ADMIN）
docker run --privileged -it myapp

# 容器内
perf top
perf record -g ./myapp
perf report
```

### 网络性能测试

```bash
# 使用 iperf3
docker run -d --name iperf-server networkstatic/iperf3 -s
docker run --rm networkstatic/iperf3 -c iperf-server

# 使用 netperf
docker run --rm -it networkstatic/netperf -H target-host
```

## 优化建议

### 镜像优化

```dockerfile
# 使用小型基础镜像
FROM alpine:latest
# 或
FROM gcr.io/distroless/static

# 多阶段构建减少镜像大小
FROM golang:1.21 AS builder
# ... build ...
FROM scratch
COPY --from=builder /app /app
```

### 启动优化

```bash
# 预拉取镜像
docker pull myapp:latest

# 使用健康检查确保服务就绪
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s \
  CMD curl -f http://localhost/health || exit 1
```

### 运行时优化

```yaml
# 合理设置资源限制
services:
  app:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 512M
    # 使用 tmpfs 加速临时文件
    tmpfs:
      - /tmp
    # 只读文件系统减少 I/O
    read_only: true
```
