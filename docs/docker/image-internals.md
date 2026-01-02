---
sidebar_position: 18
title: 镜像原理
description: Docker 镜像层原理、Union FS 与 Copy-on-Write 机制
---

# 镜像原理

深入理解 Docker 镜像的内部结构和工作原理。

## 镜像层结构

Docker 镜像由多个只读层（Layer）组成，每一层代表 Dockerfile 中的一条指令。

```
┌─────────────────────────────────────┐
│         Container Layer (R/W)       │  ← 容器可写层
├─────────────────────────────────────┤
│         Layer 4: CMD                │  ← 只读
├─────────────────────────────────────┤
│         Layer 3: COPY app           │  ← 只读
├─────────────────────────────────────┤
│         Layer 2: RUN npm install    │  ← 只读
├─────────────────────────────────────┤
│         Layer 1: FROM node:18       │  ← 只读（基础镜像）
└─────────────────────────────────────┘
```

### 查看镜像层

```bash
# 查看镜像历史（每层对应一条指令）
docker history nginx

# 详细查看镜像层
docker inspect nginx --format '{{json .RootFS.Layers}}' | jq

# 查看镜像大小分布
docker history nginx --format "{{.Size}}\t{{.CreatedBy}}"
```

## Union File System

Union FS 允许将多个目录（层）联合挂载到同一个挂载点，呈现为统一的文件系统视图。

### 存储驱动类型

| 驱动 | 说明 | 适用场景 |
|------|------|----------|
| **overlay2** | 推荐，性能最佳 | Linux 4.0+ |
| **fuse-overlayfs** | 用于 rootless 模式 | 无 root 权限 |
| **btrfs** | 需要 btrfs 文件系统 | 使用 btrfs 的系统 |
| **zfs** | 需要 zfs 文件系统 | 使用 zfs 的系统 |
| **vfs** | 无 CoW，仅用于测试 | 调试用途 |

### 查看当前存储驱动

```bash
docker info | grep "Storage Driver"
# Storage Driver: overlay2
```

### overlay2 工作原理

```
┌─────────────────────────────────────────────────┐
│              Merged View (容器看到的)            │
│  /app/index.js  /etc/nginx.conf  /var/log/...  │
└─────────────────────────────────────────────────┘
                        ▲
                        │ Union Mount
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
│   UpperDir    │ │  WorkDir  │ │   LowerDir    │
│   (可写层)     │ │  (工作区)  │ │   (只读层)    │
│ /var/lib/...  │ │           │ │ layer1+layer2 │
│ /diff         │ │           │ │    +layer3    │
└───────────────┘ └───────────┘ └───────────────┘
```

### 查看 overlay2 目录结构

```bash
# 查看镜像存储位置
ls /var/lib/docker/overlay2/

# 查看特定容器的 overlay 挂载
docker inspect container_name --format '{{json .GraphDriver.Data}}' | jq

# 输出示例
{
  "LowerDir": "/var/lib/docker/overlay2/abc.../diff:/var/lib/docker/overlay2/def.../diff",
  "MergedDir": "/var/lib/docker/overlay2/xyz.../merged",
  "UpperDir": "/var/lib/docker/overlay2/xyz.../diff",
  "WorkDir": "/var/lib/docker/overlay2/xyz.../work"
}
```

## Copy-on-Write (CoW)

CoW 机制确保只有在文件被修改时才会复制到可写层。

### CoW 工作流程

```
读取文件:
┌─────────────┐
│ 容器读取    │ → 从最上层开始查找 → 找到后直接读取
│ /etc/hosts  │
└─────────────┘

修改文件:
┌─────────────┐
│ 容器修改    │ → 从只读层复制到可写层 → 在可写层修改
│ /etc/hosts  │
└─────────────┘

删除文件:
┌─────────────┐
│ 容器删除    │ → 在可写层创建 whiteout 文件 → 遮盖下层文件
│ /etc/hosts  │
└─────────────┘
```

### 验证 CoW 行为

```bash
# 创建容器
docker run -d --name test nginx

# 查看初始可写层大小
docker ps -s
# SIZE: 0B (virtual 187MB)

# 在容器中创建文件
docker exec test dd if=/dev/zero of=/test.img bs=1M count=100

# 再次查看大小
docker ps -s
# SIZE: 105MB (virtual 292MB)

# 查看可写层内容
docker diff test
# C /
# A /test.img
```

### Whiteout 文件

删除文件时，overlay2 使用 whiteout 文件标记删除。

```bash
# 在容器中删除文件
docker exec test rm /etc/nginx/nginx.conf

# 查看变更
docker diff test
# D /etc/nginx/nginx.conf

# 在 upperdir 中会看到 whiteout 文件
# 文件名格式: .wh.<filename>
```

## 镜像内容寻址

Docker 使用内容寻址存储（Content-Addressable Storage）。

### 镜像 ID 与 Digest

```bash
# 镜像 ID（本地标识）
docker images --digests nginx
# REPOSITORY  TAG     DIGEST                                                                    IMAGE ID
# nginx       latest  sha256:abc123...                                                          def456...

# Digest 是镜像内容的 SHA256 哈希
# 相同内容的镜像在任何地方都有相同的 digest
```

### 层的内容寻址

```bash
# 查看镜像层的 digest
docker inspect nginx --format '{{json .RootFS.Layers}}' | jq
# [
#   "sha256:aaa...",
#   "sha256:bbb...",
#   "sha256:ccc..."
# ]

# 相同的层在不同镜像间共享
```

## 镜像瘦身

### 使用 dive 分析镜像

```bash
# 安装 dive
brew install dive  # macOS
# 或
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  wagoodman/dive:latest myimage:tag

# 分析镜像
dive myimage:tag
```

dive 界面说明：
- 左侧：镜像层列表和大小
- 右侧：每层的文件变更
- 底部：镜像效率评分

### 常见瘦身技巧

```dockerfile
# 1. 使用 Alpine 基础镜像
FROM node:18-alpine  # ~50MB vs node:18 ~350MB

# 2. 多阶段构建
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules

# 3. 合并 RUN 指令并清理缓存
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 4. 使用 .dockerignore
# .dockerignore
node_modules
.git
*.md
Dockerfile

# 5. 只复制必要文件
COPY package*.json ./
COPY src/ ./src/
# 而不是 COPY . .
```

### 使用 docker-slim

```bash
# 安装 docker-slim
brew install docker-slim  # macOS

# 分析镜像
docker-slim xray myimage:tag

# 自动瘦身（会运行容器进行分析）
docker-slim build myimage:tag

# 指定要保留的路径
docker-slim build --include-path=/app --include-path=/etc myimage:tag
```

### Distroless 镜像

Google 提供的最小化镜像，只包含应用运行时。

```dockerfile
# Java 应用
FROM gcr.io/distroless/java17-debian11
COPY target/app.jar /app.jar
CMD ["app.jar"]

# Node.js 应用
FROM gcr.io/distroless/nodejs18-debian11
COPY --from=builder /app/dist /app
WORKDIR /app
CMD ["index.js"]

# 静态二进制
FROM gcr.io/distroless/static-debian11
COPY myapp /
CMD ["/myapp"]
```

## 镜像层优化

### 利用构建缓存

```dockerfile
# ✅ 好的做法：依赖文件先复制
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

# ❌ 不好的做法：每次代码变更都重新安装依赖
COPY . .
RUN npm ci
```

### 减少层数

```dockerfile
# ❌ 多个 RUN 指令
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN rm -rf /var/lib/apt/lists/*

# ✅ 合并为一个 RUN
RUN apt-get update && \
    apt-get install -y curl wget && \
    rm -rf /var/lib/apt/lists/*
```

### 避免在镜像中存储敏感信息

```dockerfile
# ❌ 错误：密钥会保留在镜像层中
COPY .npmrc /root/.npmrc
RUN npm ci
RUN rm /root/.npmrc  # 删除也没用，已经在之前的层中了

# ✅ 正确：使用 BuildKit secrets
# syntax=docker/dockerfile:1
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc npm ci
```

## 镜像存储管理

```bash
# 查看镜像占用空间
docker system df
docker system df -v

# 清理悬空镜像（无标签的中间层）
docker image prune

# 清理所有未使用镜像
docker image prune -a

# 清理指定时间前的镜像
docker image prune -a --filter "until=24h"

# 导出镜像（包含所有层）
docker save myimage:tag -o myimage.tar

# 查看导出的镜像结构
tar -tvf myimage.tar
```
