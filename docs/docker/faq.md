---
sidebar_position: 10
title: 常见问题
description: Docker 常见问题与解决方案
---

# Docker 常见问题

## 安装与配置

### Docker 服务无法启动

```bash
# 查看状态
sudo systemctl status docker

# 查看详细日志
sudo journalctl -xu docker

# 常见原因：
# 1. 存储驱动问题
# 2. 配置文件语法错误
# 3. 磁盘空间不足
```

### 权限不足（permission denied）

```bash
# 方法 1：添加用户到 docker 组（推荐）
sudo usermod -aG docker $USER
newgrp docker  # 或重新登录

# 方法 2：修改 socket 权限（临时）
sudo chmod 666 /var/run/docker.sock
```

### 镜像拉取慢或超时

配置镜像加速器 `/etc/docker/daemon.json`：

```json
{
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://registry.docker-cn.com"
  ]
}
```

```bash
# 重启 Docker
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### daemon.json 配置报错

```bash
# 验证 JSON 格式
cat /etc/docker/daemon.json | python3 -m json.tool

# 常见错误：
# - 多余的逗号
# - 缺少引号
# - 不支持的选项
```

## 容器问题

### 容器启动后立即退出

```bash
# 1. 查看日志
docker logs container_name

# 2. 交互式调试
docker run -it image_name sh

# 常见原因：
# - 主进程执行完毕退出
# - 配置错误导致崩溃
# - 依赖服务未就绪
```

**解决方案**：

```dockerfile
# 确保有前台进程运行
CMD ["nginx", "-g", "daemon off;"]

# 或使用 tail 保持运行（调试用）
CMD ["tail", "-f", "/dev/null"]
```

### 无法删除容器

```bash
# 强制删除
docker rm -f container_name

# 删除所有已停止容器
docker container prune

# 删除所有容器
docker rm -f $(docker ps -aq)
```

### 容器无法连接网络

```bash
# 1. 检查 DNS
docker run --rm busybox nslookup google.com

# 2. 使用自定义 DNS
docker run --dns 8.8.8.8 myimage

# 3. 检查网络模式
docker inspect container_name | grep -i network

# 4. 重置 Docker 网络
docker network prune
sudo systemctl restart docker
```

### 容器间无法通信

```bash
# 1. 确保在同一网络
docker network create mynet
docker run --network mynet --name app1 nginx
docker run --network mynet --name app2 alpine ping app1

# 2. 使用容器名而非 IP
# 自定义网络支持 DNS 解析
redis://redis:6379  # ✅
redis://172.17.0.2:6379  # ❌ IP 可能变化
```

### 容器时区不正确

```bash
# 方法 1：挂载时区文件
docker run -v /etc/localtime:/etc/localtime:ro nginx

# 方法 2：设置环境变量
docker run -e TZ=Asia/Shanghai nginx

# 方法 3：Dockerfile 中设置
RUN apk add --no-cache tzdata && \
    cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
```

## 镜像问题

### 镜像过大

1. **使用 Alpine 基础镜像**

   ```dockerfile
   FROM node:18-alpine  # ~170MB
   # 而不是
   FROM node:18         # ~1GB
   ```

2. **多阶段构建**

   ```dockerfile
   FROM golang:1.21 AS builder
   RUN go build -o app

   FROM alpine:latest
   COPY --from=builder /go/app .
   ```

3. **清理缓存**

   ```dockerfile
   RUN apt-get update && apt-get install -y curl \
       && rm -rf /var/lib/apt/lists/*
   ```

4. **使用 .dockerignore**

### 构建失败

```bash
# 查看详细日志
docker build --progress=plain .

# 不使用缓存重新构建
docker build --no-cache .

# 常见原因：
# - 网络问题（包下载失败）
# - 依赖版本冲突
# - 权限问题
```

### 多平台构建

```bash
# 启用 BuildKit
export DOCKER_BUILDKIT=1

# 创建 builder
docker buildx create --use

# 构建多平台镜像
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myimage:latest \
  --push .
```

## 存储问题

### 磁盘空间不足

```bash
# 查看使用情况
docker system df
docker system df -v  # 详细信息

# 清理未使用资源
docker system prune       # 基本清理
docker system prune -a    # 清理所有未使用镜像
docker system prune -a --volumes  # 包括卷

# 清理特定资源
docker image prune -a --filter "until=168h"  # 7天前的镜像
docker container prune --filter "until=24h"  # 24小时前的容器
```

### 数据丢失

使用命名卷而非匿名卷：

```bash
# ❌ 匿名卷（容器删除后难以找回）
docker run -v /data mysql

# ✅ 命名卷（持久化）
docker run -v mysql-data:/var/lib/mysql mysql
```

### 卷权限问题

```bash
# 方法 1：Dockerfile 中设置权限
RUN mkdir -p /data && chown -R 1000:1000 /data
USER 1000

# 方法 2：运行时指定用户
docker run -u 1000:1000 -v myvolume:/data myimage

# 方法 3：修改主机目录权限
sudo chown -R 1000:1000 /host/path
```

## 网络问题

### 端口冲突

```bash
# 查看端口占用
lsof -i :80
netstat -tlnp | grep 80

# 使用其他端口
docker run -p 8080:80 nginx

# 随机端口
docker run -P nginx
docker port container_name
```

### 无法访问容器服务

```bash
# 1. 检查端口映射
docker port container_name

# 2. 检查容器是否运行
docker ps

# 3. 检查服务是否启动
docker exec container_name curl localhost:80

# 4. 检查防火墙
sudo iptables -L -n
sudo ufw status
```

## 性能问题

### 容器 CPU 使用率高

```bash
# 查看资源使用
docker stats container_name

# 查看进程
docker top container_name

# 进入容器排查
docker exec -it container_name top

# 限制 CPU
docker update --cpus 1.0 container_name
```

### 容器内存不足

```bash
# 查看内存使用
docker stats --format "table {{.Name}}\t{{.MemUsage}}"

# 增加内存限制
docker update --memory 1g --memory-swap 2g container_name

# 分析内存使用
docker exec container_name cat /proc/meminfo
```

### I/O 性能差（macOS/Windows）

```yaml
# 使用缓存模式优化
services:
  app:
    volumes:
      - ./src:/app/src:cached # 优化读取
      - ./data:/app/data:delegated # 优化写入
```

## Docker Compose 问题

### Compose 与 Swarm 区别

| 特性 | Docker Compose     | Docker Swarm           |
| ---- | ------------------ | ---------------------- |
| 用途 | 单机编排           | 多机集群               |
| 配置 | docker-compose.yml | 相同（加 deploy 配置） |
| 命令 | `docker compose`   | `docker stack`         |
| 扩展 | 单机多副本         | 跨节点分布             |
| 适用 | 开发、测试         | 生产                   |

### 服务启动顺序问题

`depends_on` 只保证启动顺序，不保证服务就绪：

```yaml
services:
  app:
    depends_on:
      db:
        condition: service_healthy # 等待健康检查通过

  db:
    image: postgres
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### 环境变量未生效

```bash
# 1. 检查 .env 文件位置（必须在 docker-compose.yml 同目录）

# 2. 检查变量语法
environment:
  - VAR_NAME=value    # 正确
  - VAR_NAME=${VAR}   # 从 .env 读取

# 3. 检查 env_file
env_file:
  - .env
  - .env.local
```

## 私有仓库

### 配置私有仓库

```bash
# 登录
docker login registry.example.com

# 推送
docker tag myimage registry.example.com/myimage:v1
docker push registry.example.com/myimage:v1

# 拉取
docker pull registry.example.com/myimage:v1
```

### 配置不安全仓库

```json
// /etc/docker/daemon.json
{
  "insecure-registries": ["registry.example.com:5000"]
}
```

### 搭建私有仓库

```bash
docker run -d \
  --name registry \
  -p 5000:5000 \
  -v registry-data:/var/lib/registry \
  registry:2
```

## 日志问题

### 日志文件过大

```json
// /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

### 查看日志位置

```bash
# 日志存储位置
/var/lib/docker/containers/<container-id>/<container-id>-json.log

# 清空日志（谨慎使用）
truncate -s 0 /var/lib/docker/containers/<container-id>/*-json.log
```
