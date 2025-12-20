---
sidebar_position: 5
title: BuildKit 与 buildx
description: 使用 BuildKit/buildx 提升 Docker 构建速度、复用缓存与多平台构建
---

# BuildKit 与 buildx

## BuildKit 是什么

BuildKit 是 Docker 新一代构建引擎，主要带来：

- 更快的并行构建
- 更强的缓存能力（本地/远端）
- 更安全的构建能力（`secret`/`ssh` mount，不落镜像层）
- 更好的多平台构建支持（配合 `buildx`）

## 启用 BuildKit

一次性启用：

```bash
DOCKER_BUILDKIT=1 docker build .
```

建议在 CI 中默认启用，并使用：

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain .
```

以获得更可读的日志。

## buildx 入门

buildx 是 Docker 的构建扩展（通常已随 Docker Desktop / 新版 Docker 安装）。

```bash
# 查看 buildx 版本与当前 builder
docker buildx version

docker buildx ls

# 创建并使用新的 builder（常用于多平台）
docker buildx create --use --name mybuilder
```

## Dockerfile 前置语法（可选但常用）

当你需要使用 BuildKit 的高级特性时，建议在 Dockerfile 顶部指定语法：

```dockerfile
# syntax=docker/dockerfile:1.6
```

## 高级缓存：cache mount

典型场景：包管理器缓存（npm/pip/maven/gradle）不想每次都重新下载。

```dockerfile
# syntax=docker/dockerfile:1.6
FROM node:18-alpine
WORKDIR /app

COPY package*.json ./

RUN --mount=type=cache,target=/root/.npm \
    npm ci

COPY . .
RUN npm run build
```

## 构建时 secret（不落镜像层）

适用：私有依赖、私有 npm registry、云密钥等。

Dockerfile：

```dockerfile
# syntax=docker/dockerfile:1.6
FROM node:18-alpine
WORKDIR /app

COPY package*.json ./

RUN --mount=type=secret,id=npmrc,target=/root/.npmrc \
    npm ci
```

构建命令：

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=npmrc,src=.npmrc \
  -t myapp:buildkit .
```

## SSH mount（拉私有 git 依赖）

```dockerfile
# syntax=docker/dockerfile:1.6
FROM alpine:3.19
RUN apk add --no-cache git openssh

RUN --mount=type=ssh \
    git clone git@github.com:org/private-repo.git /src
```

构建命令：

```bash
DOCKER_BUILDKIT=1 docker build --ssh default .
```

## 多平台构建（amd64/arm64）

```bash
# 构建并推送多平台镜像
# 注意：--push 会直接推到仓库（CI 常用）
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t registry.example.com/team/app:v1.0.0 \
  --push \
  .
```

只在本地验证（不推送）时常用：

```bash
# 将单平台结果加载到本地 docker images
# 注意：多平台无法 --load，只能 --push 或输出到本地目录
docker buildx build \
  --platform linux/amd64 \
  -t app:local \
  --load \
  .
```

## 远端缓存（CI 加速）

CI 中建议把缓存推到镜像仓库或缓存后端。

```bash
docker buildx build \
  --cache-to type=registry,ref=registry.example.com/team/app:buildcache,mode=max \
  --cache-from type=registry,ref=registry.example.com/team/app:buildcache \
  -t registry.example.com/team/app:v1.0.0 \
  --push \
  .
```

## 常见坑

- **`.dockerignore` 排除了构建所需文件**：导致 `COPY` 失败或构建产物缺失
- **平台不一致**：在 Apple Silicon 上默认可能是 `arm64`，生产可能是 `amd64`
- **secret 写进镜像层**：不要用 `ARG`/`ENV` 保存密钥，优先 `--secret`

## 排错

```bash
# 输出更详细构建日志
DOCKER_BUILDKIT=1 docker build --progress=plain .

# 查看 buildx builder
Docker buildx ls
```
