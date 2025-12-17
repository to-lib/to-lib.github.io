---
sidebar_position: 8
title: 最佳实践
description: Docker 生产环境最佳实践指南
---

# Docker 最佳实践

本文介绍 Docker 在生产环境中的最佳实践。

## 镜像构建

### 使用官方基础镜像

```dockerfile
# ✅ 推荐
FROM node:18-alpine
FROM nginx:1.25-alpine

# ❌ 避免
FROM ubuntu:latest
```

### 指定精确版本

```dockerfile
# ✅ 推荐：精确版本
FROM node:18.19.0-alpine3.19

# ⚠️ 可接受：主版本
FROM node:18-alpine

# ❌ 避免：latest 标签
FROM node:latest
```

### 使用多阶段构建

```dockerfile
# 构建阶段
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# 生产阶段
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
USER node
CMD ["node", "app.js"]
```

### 合并 RUN 指令

```dockerfile
# ✅ 推荐：减少层数
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        wget && \
    rm -rf /var/lib/apt/lists/*

# ❌ 避免：多个 RUN
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
```

### 使用 .dockerignore

```plaintext
# .dockerignore
.git
.gitignore
node_modules
npm-debug.log
*.md
Dockerfile*
docker-compose*
.env*
.vscode
.idea
coverage
tests
```

### 利用构建缓存

```dockerfile
# ✅ 推荐：依赖文件先复制
COPY package.json package-lock.json ./
RUN npm ci

# 源码后复制（变化频繁）
COPY . .
RUN npm run build
```

## 安全性

### 使用非 root 用户

```dockerfile
# 创建专用用户
RUN groupadd -r app && useradd -r -g app app

# 设置目录权限
RUN chown -R app:app /app

# 切换用户
USER app
```

### 最小化镜像

```dockerfile
# 按大小排序（小 → 大）
FROM scratch                           # 最小
FROM gcr.io/distroless/static         # ~2MB
FROM gcr.io/distroless/nodejs18       # ~130MB
FROM node:18-alpine                   # ~170MB
FROM node:18-slim                     # ~240MB
FROM node:18                          # ~1GB
```

### 资源限制

```bash
docker run -d \
  --name myapp \
  --memory=512m \
  --memory-swap=512m \
  --cpus=1.0 \
  --pids-limit=100 \
  myapp:latest
```

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
        reservations:
          cpus: "0.5"
          memory: 256M
```

### 只读文件系统

```bash
docker run --read-only \
  --tmpfs /tmp \
  --tmpfs /var/run \
  nginx
```

### 禁用特权

```bash
# ❌ 绝对禁止
docker run --privileged myapp

# ✅ 仅授予必需权限
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE myapp
```

## 健康检查

### Dockerfile 配置

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
  CMD curl -f http://localhost:8080/health || exit 1
```

### Compose 配置

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

### 数据库健康检查

```yaml
services:
  mysql:
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
```

## 日志管理

### 配置日志驱动

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

### 容器级配置

```yaml
services:
  app:
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

## 网络安全

### 隔离敏感服务

```yaml
networks:
  frontend:
  backend:
    internal: true # 无法访问外部网络

services:
  web:
    networks: [frontend]
  app:
    networks: [frontend, backend]
  db:
    networks: [backend]
```

### 仅暴露必要端口

```yaml
services:
  app:
    expose:
      - "3000" # 仅内部访问

  nginx:
    ports:
      - "80:80" # 外部访问
```

## CI/CD 集成

### 镜像标签策略

```bash
# 推荐的标签方案
myapp:v1.2.3                    # 语义化版本
myapp:v1.2.3-abc1234            # 版本 + Git SHA
myapp:latest                     # 最新稳定版
myapp:develop                    # 开发分支

# 构建示例
VERSION=$(git describe --tags)
COMMIT=$(git rev-parse --short HEAD)
docker build -t myapp:${VERSION} -t myapp:${VERSION}-${COMMIT} .
```

### GitHub Actions 示例

```yaml
# .github/workflows/docker.yml
name: Docker Build

on:
  push:
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Docker meta
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=semver,pattern={{version}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: ${{ steps.meta.outputs.tags }}
```

## 容器编排

### 重启策略

```yaml
services:
  app:
    restart: unless-stopped # 推荐
    # restart: always        # 始终重启
    # restart: on-failure    # 仅失败时重启
    # restart: no            # 不重启
```

### 服务依赖

```yaml
services:
  app:
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
```

### 滚动更新

```yaml
services:
  app:
    deploy:
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
        order: start-first
      rollback_config:
        parallelism: 1
        delay: 10s
```

## 资源管理

### 定期清理

```bash
# 清理所有未使用资源
docker system prune -a --volumes

# 自动化清理脚本
#!/bin/bash
docker image prune -a --filter "until=168h" -f  # 7天前的镜像
docker container prune --filter "until=24h" -f  # 24小时前的容器
docker volume prune -f
docker network prune -f
```

### 磁盘限制

```json
// /etc/docker/daemon.json
{
  "storage-driver": "overlay2",
  "storage-opts": ["overlay2.size=20G"]
}
```

## 故障恢复

### 备份策略

```bash
# 备份卷数据
docker run --rm \
  -v mydata:/data \
  -v $(pwd)/backup:/backup \
  alpine tar cvf /backup/mydata-$(date +%Y%m%d).tar /data

# 备份数据库
docker exec mysql mysqldump -u root -p$MYSQL_ROOT_PASSWORD mydb > backup.sql
```

### 恢复策略

```bash
# 恢复卷数据
docker run --rm \
  -v mydata:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xvf /backup/mydata-20240101.tar -C /

# 恢复数据库
docker exec -i mysql mysql -u root -p$MYSQL_ROOT_PASSWORD mydb < backup.sql
```

## 检查清单

### 镜像构建

- [ ] 使用官方基础镜像
- [ ] 指定精确版本
- [ ] 多阶段构建
- [ ] 合并 RUN 指令
- [ ] 使用 .dockerignore
- [ ] 清理缓存和临时文件

### 安全

- [ ] 使用非 root 用户
- [ ] 最小化镜像
- [ ] 设置资源限制
- [ ] 只读文件系统（如可能）
- [ ] 禁用特权模式
- [ ] 删除不必要的 capabilities

### 运行时

- [ ] 配置健康检查
- [ ] 设置日志限制
- [ ] 网络隔离
- [ ] 仅暴露必要端口
- [ ] 配置重启策略

### 运维

- [ ] 建立镜像标签规范
- [ ] CI/CD 集成
- [ ] 定期清理资源
- [ ] 备份策略
- [ ] 监控告警
