---
sidebar_position: 8
title: 最佳实践
description: Docker 生产环境最佳实践指南
---

# Docker 最佳实践

## 镜像构建

### 使用官方基础镜像

```dockerfile
# ✅ 推荐
FROM node:18-alpine

# ❌ 避免
FROM ubuntu:latest
```

### 指定精确版本

```dockerfile
# ✅ 推荐
FROM node:18.19.0-alpine3.19

# ❌ 避免
FROM node:latest
```

### 使用多阶段构建

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine
COPY --from=builder /app/node_modules ./node_modules
COPY . .
CMD ["node", "app.js"]
```

### 合并 RUN 指令

```dockerfile
# ✅ 推荐
RUN apt-get update && \
    apt-get install -y curl wget && \
    rm -rf /var/lib/apt/lists/*
```

### 使用 .dockerignore

```plaintext
.git
node_modules
*.md
Dockerfile
.env
```

## 安全性

### 使用非 root 用户

```dockerfile
RUN groupadd -r app && useradd -r -g app app
USER app
```

### 最小化镜像

```dockerfile
FROM node:18-alpine  # 小镜像
FROM gcr.io/distroless/nodejs18  # 更小
```

### 限制资源

```bash
docker run --memory="512m" --cpus="1.0" myapp
```

## 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8080/health || exit 1
```

## 日志管理

```json
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "10m", "max-file": "3" }
}
```

## 网络安全

```yaml
networks:
  backend:
    internal: true # 无法访问外部
```

## 检查清单

- [ ] 使用精确版本
- [ ] 多阶段构建
- [ ] 非 root 用户
- [ ] 设置资源限制
- [ ] 配置健康检查
