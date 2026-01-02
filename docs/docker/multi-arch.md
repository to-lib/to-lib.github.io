---
sidebar_position: 19
title: 多架构镜像
description: 使用 Docker Buildx 构建跨平台镜像 (amd64/arm64)
---

# 多架构镜像

构建支持多种 CPU 架构的 Docker 镜像，实现一次构建、多平台运行。

## 为什么需要多架构镜像

- **Apple Silicon (M1/M2/M3)** - arm64 架构
- **AWS Graviton** - arm64 架构，性价比更高
- **树莓派** - arm/arm64 架构
- **传统服务器** - amd64 架构

## Docker Buildx

Buildx 是 Docker 的扩展构建工具，支持多平台构建。

### 检查 Buildx

```bash
# 查看 buildx 版本
docker buildx version

# 列出构建器
docker buildx ls

# 默认输出
NAME/NODE       DRIVER/ENDPOINT STATUS  BUILDKIT PLATFORMS
default         docker
  default       default         running 23.0.1   linux/amd64, linux/arm64
```

### 创建多平台构建器

```bash
# 创建新的构建器实例
docker buildx create --name multiarch --driver docker-container --bootstrap

# 使用该构建器
docker buildx use multiarch

# 查看支持的平台
docker buildx inspect --bootstrap
```

## 构建多架构镜像

### 基本构建

```bash
# 构建并推送多架构镜像
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myregistry/myapp:latest \
  --push \
  .

# 构建并加载到本地（仅支持单平台）
docker buildx build \
  --platform linux/arm64 \
  -t myapp:latest \
  --load \
  .
```

### 构建并导出

```bash
# 导出为 OCI 格式
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myapp:latest \
  --output type=oci,dest=myapp.tar \
  .

# 导出为 Docker 格式（单平台）
docker buildx build \
  --platform linux/amd64 \
  -t myapp:latest \
  --output type=docker,dest=myapp-amd64.tar \
  .
```

## Dockerfile 多架构适配

### 使用 ARG 获取平台信息

```dockerfile
FROM --platform=$BUILDPLATFORM golang:1.21 AS builder

# 构建平台信息
ARG BUILDPLATFORM
ARG BUILDOS
ARG BUILDARCH

# 目标平台信息
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

WORKDIR /app
COPY . .

# 交叉编译
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -o /app/main .

FROM alpine:latest
COPY --from=builder /app/main /main
CMD ["/main"]
```

### 平台特定基础镜像

```dockerfile
# 自动选择对应架构的基础镜像
FROM node:18-alpine

# 或明确指定
FROM --platform=linux/amd64 node:18-alpine
```

### 条件安装依赖

```dockerfile
FROM ubuntu:22.04

ARG TARGETARCH

# 根据架构安装不同的包
RUN apt-get update && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      apt-get install -y package-arm64; \
    else \
      apt-get install -y package-amd64; \
    fi
```

## 实战示例

### Go 应用

```dockerfile
FROM --platform=$BUILDPLATFORM golang:1.21-alpine AS builder

ARG TARGETOS TARGETARCH

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .

RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -ldflags="-s -w" -o /app/server .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
COPY --from=builder /app/server /server
EXPOSE 8080
CMD ["/server"]
```

### Node.js 应用

```dockerfile
FROM --platform=$BUILDPLATFORM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["node", "dist/main.js"]
```

### Rust 应用

```dockerfile
FROM --platform=$BUILDPLATFORM rust:1.74 AS builder

ARG TARGETPLATFORM
RUN case "$TARGETPLATFORM" in \
      "linux/amd64") echo "x86_64-unknown-linux-musl" > /target.txt ;; \
      "linux/arm64") echo "aarch64-unknown-linux-musl" > /target.txt ;; \
    esac

RUN rustup target add $(cat /target.txt)

WORKDIR /app
COPY . .
RUN cargo build --release --target $(cat /target.txt) && \
    cp target/$(cat /target.txt)/release/myapp /myapp

FROM alpine:latest
COPY --from=builder /myapp /myapp
CMD ["/myapp"]
```

## Manifest 操作

### 查看多架构镜像

```bash
# 查看镜像支持的架构
docker manifest inspect nginx:latest

# 简洁输出
docker manifest inspect --verbose nginx:latest | jq '.[].Platform'
```

### 手动创建 Manifest

```bash
# 分别构建各架构镜像
docker build -t myapp:amd64 --platform linux/amd64 .
docker build -t myapp:arm64 --platform linux/arm64 .

# 推送各架构镜像
docker push myapp:amd64
docker push myapp:arm64

# 创建并推送 manifest
docker manifest create myapp:latest \
  myapp:amd64 \
  myapp:arm64

docker manifest push myapp:latest
```

## QEMU 模拟

在非原生架构上构建时，Docker 使用 QEMU 进行模拟。

### 安装 QEMU

```bash
# 安装 QEMU 用户态模拟器
docker run --privileged --rm tonistiigi/binfmt --install all

# 验证安装
docker run --rm --platform linux/arm64 alpine uname -m
# aarch64
```

### 性能考虑

| 方式 | 速度 | 适用场景 |
|------|------|----------|
| 原生构建 | 最快 | 有对应架构的机器 |
| 交叉编译 | 快 | Go、Rust 等支持交叉编译的语言 |
| QEMU 模拟 | 慢 | 无法交叉编译时的备选方案 |

## CI/CD 集成

### GitHub Actions

```yaml
name: Build Multi-Arch Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and Push
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### GitLab CI

```yaml
build:
  image: docker:24
  services:
    - docker:24-dind
  variables:
    DOCKER_BUILDKIT: 1
  before_script:
    - docker buildx create --use
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker buildx build
        --platform linux/amd64,linux/arm64
        --push
        -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
        .
```

## 常见问题

### 构建缓存

```bash
# 使用 registry 缓存
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --cache-from type=registry,ref=myregistry/myapp:cache \
  --cache-to type=registry,ref=myregistry/myapp:cache,mode=max \
  -t myregistry/myapp:latest \
  --push \
  .
```

### 调试构建

```bash
# 查看构建日志
docker buildx build --progress=plain ...

# 只构建特定阶段
docker buildx build --target builder ...
```
