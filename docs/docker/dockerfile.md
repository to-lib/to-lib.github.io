---
sidebar_position: 4
title: Dockerfile 编写
description: Dockerfile 指令详解与最佳实践
---

# Dockerfile 编写

Dockerfile 是一个文本文件，包含了构建 Docker 镜像所需的所有指令。

## 基本结构

```dockerfile
# 基础镜像
FROM ubuntu:22.04

# 维护者信息
LABEL maintainer="developer@example.com"

# 设置环境变量
ENV APP_HOME=/app

# 设置工作目录
WORKDIR $APP_HOME

# 复制文件
COPY . .

# 运行命令
RUN apt-get update && apt-get install -y nginx

# 暴露端口
EXPOSE 80

# 启动命令
CMD ["nginx", "-g", "daemon off;"]
```

## 核心指令

### FROM - 基础镜像

```dockerfile
# 使用官方镜像
FROM nginx:latest

# 使用特定版本
FROM node:18-alpine

# 使用多阶段构建
FROM golang:1.21 AS builder
FROM alpine:latest
```

### RUN - 执行命令

```dockerfile
# Shell 形式
RUN apt-get update && apt-get install -y curl

# Exec 形式
RUN ["apt-get", "update"]

# 多行命令（推荐）
RUN apt-get update && \
    apt-get install -y \
        curl \
        wget \
        vim && \
    rm -rf /var/lib/apt/lists/*
```

### COPY 和 ADD

```dockerfile
# COPY - 简单复制
COPY src/ /app/src/
COPY package.json /app/

# COPY 使用 --chown
COPY --chown=user:group files* /app/

# ADD - 支持 URL 和解压
ADD https://example.com/file.tar.gz /app/
ADD archive.tar.gz /app/
```

:::tip
优先使用 `COPY`，只有在需要解压或从 URL 下载时才使用 `ADD`。
:::

### WORKDIR - 工作目录

```dockerfile
WORKDIR /app
WORKDIR src  # 相对路径，结果为 /app/src
```

### ENV - 环境变量

```dockerfile
# 单个变量
ENV NODE_ENV=production

# 多个变量
ENV NODE_ENV=production \
    PORT=3000 \
    DB_HOST=localhost
```

### ARG - 构建参数

```dockerfile
ARG VERSION=latest
ARG BUILD_DATE

FROM node:${VERSION}

# 使用构建参数
RUN echo "Build date: ${BUILD_DATE}"
```

构建时传递参数：

```bash
docker build --build-arg VERSION=18 --build-arg BUILD_DATE=$(date +%Y%m%d) .
```

### EXPOSE - 暴露端口

```dockerfile
EXPOSE 80
EXPOSE 80/tcp
EXPOSE 443/tcp 8080/tcp
```

### CMD 和 ENTRYPOINT

```dockerfile
# CMD - 默认命令（可被覆盖）
CMD ["nginx", "-g", "daemon off;"]
CMD nginx -g "daemon off;"

# ENTRYPOINT - 入口点（不易被覆盖）
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
```

组合使用：

```dockerfile
ENTRYPOINT ["python", "app.py"]
CMD ["--port", "8080"]  # 可被覆盖的默认参数
```

### VOLUME - 数据卷

```dockerfile
VOLUME /data
VOLUME ["/data", "/logs"]
```

### USER - 运行用户

```dockerfile
# 创建用户并切换
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
USER appuser

# 使用 UID
USER 1000
```

### HEALTHCHECK - 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# 禁用健康检查
HEALTHCHECK NONE
```

## 多阶段构建

### Go 应用示例

```dockerfile
# 构建阶段
FROM golang:1.21 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o main .

# 运行阶段
FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/main .
CMD ["./main"]
```

### Node.js 应用示例

```dockerfile
# 构建阶段
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 运行阶段
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["node", "dist/main.js"]
```

### Java 应用示例

```dockerfile
# 构建阶段
FROM maven:3.9-eclipse-temurin-17 AS builder
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline
COPY src ./src
RUN mvn package -DskipTests

# 运行阶段
FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY --from=builder /app/target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
```

## 最佳实践

### 1. 使用 .dockerignore

```plaintext
# .dockerignore
node_modules
npm-debug.log
.git
.gitignore
*.md
Dockerfile
.dockerignore
.env
```

### 2. 合并 RUN 指令

```dockerfile
# ❌ 不推荐
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget

# ✅ 推荐
RUN apt-get update && \
    apt-get install -y curl wget && \
    rm -rf /var/lib/apt/lists/*
```

### 3. 利用构建缓存

```dockerfile
# ✅ 先复制依赖文件
COPY package.json package-lock.json ./
RUN npm ci

# 再复制源代码（变化频繁）
COPY . .
```

### 4. 使用非 root 用户

```dockerfile
RUN groupadd -r app && useradd -r -g app app
USER app
```

### 5. 指定确切版本

```dockerfile
# ❌ 不推荐
FROM node:latest

# ✅ 推荐
FROM node:18.19.0-alpine3.19
```

### 6. 清理缓存

```dockerfile
RUN apt-get update && \
    apt-get install -y nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

## 构建镜像

```bash
# 基本构建
docker build -t myapp:v1 .

# 指定 Dockerfile
docker build -f Dockerfile.prod -t myapp:prod .

# 不使用缓存
docker build --no-cache -t myapp:v1 .

# 多平台构建
docker buildx build --platform linux/amd64,linux/arm64 -t myapp:v1 .
```
