---
sidebar_position: 8
title: 镜像构建
description: Podman 镜像构建与 Buildah 集成
---

# Podman 镜像构建

Podman 提供与 Docker 兼容的镜像构建功能，同时支持 Buildah 进行更灵活的构建。

## 基础构建

### 使用 Dockerfile

```bash
# 构建镜像
podman build -t myapp:v1 .

# 指定 Dockerfile
podman build -f Dockerfile.prod -t myapp:prod .

# 构建时传递参数
podman build --build-arg VERSION=1.0 -t myapp .

# 无缓存构建
podman build --no-cache -t myapp .
```

### Dockerfile 示例

```dockerfile
FROM alpine:3.18

# 安装依赖
RUN apk add --no-cache python3 py3-pip

# 设置工作目录
WORKDIR /app

# 复制文件
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

# 设置环境变量
ENV PORT=8080

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["python3", "app.py"]
```

## 多阶段构建

减小镜像体积，提高安全性：

```dockerfile
# 构建阶段
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o myapp

# 运行阶段
FROM alpine:3.18
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/myapp /usr/local/bin/
CMD ["myapp"]
```

```bash
# 只构建特定阶段
podman build --target builder -t myapp:builder .
```

## Buildah 集成

Buildah 是 Podman 的底层镜像构建工具，提供更细粒度的控制。

### 安装 Buildah

```bash
# Fedora/RHEL
sudo dnf install buildah

# Ubuntu/Debian
sudo apt install buildah
```

### 脚本式构建

```bash
#!/bin/bash
# 创建容器
ctr=$(buildah from alpine:3.18)

# 运行命令
buildah run $ctr apk add --no-cache python3

# 复制文件
buildah copy $ctr ./app /app

# 设置配置
buildah config --workingdir /app $ctr
buildah config --cmd "python3 app.py" $ctr
buildah config --port 8080 $ctr

# 提交镜像
buildah commit $ctr myapp:v1

# 清理
buildah rm $ctr
```

### Buildah 常用命令

```bash
# 从基础镜像创建容器
buildah from alpine

# 挂载容器文件系统
mnt=$(buildah mount $container)
echo "mounted at $mnt"

# 卸载
buildah unmount $container

# 查看构建中的容器
buildah containers

# 删除构建容器
buildah rm $container
```

## Skopeo 镜像操作

Skopeo 用于镜像的复制、检查和签名。

### 安装 Skopeo

```bash
# Fedora/RHEL
sudo dnf install skopeo

# Ubuntu/Debian
sudo apt install skopeo
```

### 常用操作

```bash
# 检查远程镜像
skopeo inspect docker://nginx:alpine

# 复制镜像
skopeo copy docker://nginx:alpine docker://myregistry.com/nginx:alpine

# 复制到本地目录
skopeo copy docker://nginx:alpine dir:/tmp/nginx

# 复制到 Podman 本地存储
skopeo copy docker://nginx:alpine containers-storage:nginx:alpine

# 复制并更改格式
skopeo copy docker://nginx:alpine oci:/tmp/nginx-oci

# 同步镜像仓库
skopeo sync --src docker --dest docker docker.io/library/nginx myregistry.com/nginx
```

## 镜像管理

### 推送到仓库

```bash
# 登录仓库
podman login registry.example.com

# 标记镜像
podman tag myapp:v1 registry.example.com/myapp:v1

# 推送
podman push registry.example.com/myapp:v1

# 推送所有标签
podman push --all-tags registry.example.com/myapp
```

### 私有仓库配置

编辑 `~/.config/containers/registries.conf`：

```toml
[[registry]]
location = "registry.example.com"
insecure = false

[[registry.mirror]]
location = "mirror.example.com"
```

### 镜像保存与加载

```bash
# 保存镜像到文件
podman save -o myapp.tar myapp:v1

# 保存为压缩格式
podman save myapp:v1 | gzip > myapp.tar.gz

# 加载镜像
podman load -i myapp.tar

# 从标准输入加载
gunzip -c myapp.tar.gz | podman load
```

## 构建优化

### 缓存优化

```dockerfile
# 利用构建缓存
# 将不常变化的层放在前面

FROM node:18-alpine

# 先复制依赖文件
COPY package*.json ./
RUN npm install

# 再复制源代码
COPY . .

RUN npm run build
```

### 减小镜像体积

```dockerfile
# 使用 alpine 基础镜像
FROM alpine:3.18

# 合并 RUN 指令
RUN apk add --no-cache \
    python3 \
    py3-pip \
    && pip3 install --no-cache-dir flask \
    && rm -rf /var/cache/apk/*

# 使用 .dockerignore 排除不需要的文件
```

### .containerignore 文件

```plaintext
# .containerignore (或 .dockerignore)
.git
.gitignore
*.md
Dockerfile*
docker-compose*.yml
.env*
node_modules
__pycache__
*.pyc
.pytest_cache
```

## 构建缓存管理

```bash
# 查看构建缓存
podman system df

# 清理构建缓存
podman builder prune

# 清理所有未使用的数据
podman system prune -a
```

## CI/CD 集成

### GitHub Actions 示例

```yaml
name: Build and Push

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: |
          podman build -t myapp:${{ github.sha }} .

      - name: Push to registry
        run: |
          podman login -u ${{ secrets.REGISTRY_USER }} -p ${{ secrets.REGISTRY_PASS }} registry.example.com
          podman push myapp:${{ github.sha }} registry.example.com/myapp:${{ github.sha }}
```

## 最佳实践

1. **使用多阶段构建** 减小最终镜像体积
2. **合理利用构建缓存**，将变化少的层放在前面
3. **使用 .containerignore** 排除不需要的文件
4. **选择合适的基础镜像**（alpine、distroless 等）
5. **使用 Buildah** 进行复杂的构建流程
6. **使用 Skopeo** 进行跨仓库镜像同步
