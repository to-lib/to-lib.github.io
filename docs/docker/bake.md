---
sidebar_position: 34
title: Docker Bake
description: BuildKit 高级构建配置 - 使用 HCL 定义复杂构建
---

# Docker Bake

Docker Bake 是 BuildKit 的高级构建工具，使用 HCL（HashiCorp Configuration Language）或 JSON 定义复杂的多目标构建。

## 为什么使用 Bake

- **多目标构建** - 一次定义，构建多个镜像
- **变量和函数** - 动态配置构建参数
- **矩阵构建** - 自动生成多平台/多版本组合
- **继承和组合** - 复用构建配置

## 基本配置

### docker-bake.hcl

```hcl
# 变量定义
variable "TAG" {
  default = "latest"
}

variable "REGISTRY" {
  default = "docker.io/myuser"
}

# 构建目标
target "app" {
  dockerfile = "Dockerfile"
  context    = "."
  tags       = ["${REGISTRY}/myapp:${TAG}"]
}
```

### 运行构建

```bash
# 构建默认目标
docker buildx bake

# 构建指定目标
docker buildx bake app

# 传递变量
docker buildx bake --set TAG=v1.0.0

# 使用环境变量
TAG=v1.0.0 docker buildx bake

# 只打印配置，不构建
docker buildx bake --print
```

## 多目标构建

```hcl
variable "TAG" {
  default = "latest"
}

# 组：同时构建多个目标
group "default" {
  targets = ["frontend", "backend", "worker"]
}

target "frontend" {
  dockerfile = "frontend/Dockerfile"
  context    = "frontend"
  tags       = ["myapp/frontend:${TAG}"]
}

target "backend" {
  dockerfile = "backend/Dockerfile"
  context    = "backend"
  tags       = ["myapp/backend:${TAG}"]
}

target "worker" {
  dockerfile = "worker/Dockerfile"
  context    = "worker"
  tags       = ["myapp/worker:${TAG}"]
}
```

```bash
# 构建所有目标
docker buildx bake

# 只构建 frontend 和 backend
docker buildx bake frontend backend
```

## 目标继承

```hcl
# 基础目标（不会被直接构建）
target "_base" {
  dockerfile = "Dockerfile"
  args = {
    NODE_VERSION = "18"
  }
}

# 开发环境
target "dev" {
  inherits = ["_base"]
  target   = "development"
  tags     = ["myapp:dev"]
}

# 生产环境
target "prod" {
  inherits = ["_base"]
  target   = "production"
  tags     = ["myapp:latest", "myapp:${TAG}"]
  platforms = ["linux/amd64", "linux/arm64"]
}
```

## 多平台构建

```hcl
target "multiarch" {
  dockerfile = "Dockerfile"
  platforms  = [
    "linux/amd64",
    "linux/arm64",
    "linux/arm/v7"
  ]
  tags = ["myapp:latest"]
}
```

## 矩阵构建

```hcl
variable "GO_VERSIONS" {
  default = ["1.20", "1.21", "1.22"]
}

# 使用函数生成多个目标
target "go-matrix" {
  name       = "go-${replace(go_version, ".", "-")}"
  matrix     = {
    go_version = GO_VERSIONS
  }
  dockerfile = "Dockerfile"
  args = {
    GO_VERSION = go_version
  }
  tags = ["myapp:go${go_version}"]
}
```

## 内置函数

```hcl
variable "TAG" {
  default = "latest"
}

target "app" {
  tags = [
    # 字符串函数
    "myapp:${lower(TAG)}",
    "myapp:${upper(TAG)}",
    "myapp:${replace(TAG, ".", "-")}",
    
    # 时间函数
    "myapp:${formatdate("YYYYMMDD", timestamp())}",
  ]
  
  labels = {
    # 获取 Git 信息（需要在 Git 仓库中）
    "org.opencontainers.image.revision" = "${BAKE_GIT_SHA}"
  }
}

# 条件表达式
target "conditional" {
  platforms = TAG == "latest" ? ["linux/amd64", "linux/arm64"] : ["linux/amd64"]
}
```

## 缓存配置

```hcl
target "app" {
  dockerfile = "Dockerfile"
  tags       = ["myapp:latest"]
  
  # 使用 registry 缓存
  cache-from = ["type=registry,ref=myapp:cache"]
  cache-to   = ["type=registry,ref=myapp:cache,mode=max"]
  
  # 或使用本地缓存
  # cache-from = ["type=local,src=/tmp/.buildx-cache"]
  # cache-to   = ["type=local,dest=/tmp/.buildx-cache-new,mode=max"]
}
```

## 输出配置

```hcl
target "app" {
  dockerfile = "Dockerfile"
  
  # 推送到 registry
  output = ["type=registry"]
  
  # 或导出为 tar
  # output = ["type=tar,dest=./image.tar"]
  
  # 或加载到本地 Docker
  # output = ["type=docker"]
}
```

## CI/CD 集成

### GitHub Actions

```yaml
- name: Build with Bake
  uses: docker/bake-action@v4
  with:
    targets: prod
    push: true
    set: |
      *.cache-from=type=gha
      *.cache-to=type=gha,mode=max
```

### 环境变量

```hcl
variable "CI" {
  default = false
}

target "app" {
  output = CI ? ["type=registry"] : ["type=docker"]
}
```

```bash
CI=true docker buildx bake
```

## 完整示例

```hcl
# docker-bake.hcl
variable "REGISTRY" {
  default = "ghcr.io/myorg"
}

variable "TAG" {
  default = "latest"
}

variable "PLATFORMS" {
  default = ["linux/amd64", "linux/arm64"]
}

group "default" {
  targets = ["app"]
}

group "all" {
  targets = ["app", "app-debug"]
}

target "_common" {
  dockerfile = "Dockerfile"
  context    = "."
  labels = {
    "org.opencontainers.image.source" = "https://github.com/myorg/myapp"
  }
}

target "app" {
  inherits  = ["_common"]
  target    = "production"
  platforms = PLATFORMS
  tags = [
    "${REGISTRY}/myapp:${TAG}",
    "${REGISTRY}/myapp:latest"
  ]
  cache-from = ["type=registry,ref=${REGISTRY}/myapp:cache"]
  cache-to   = ["type=registry,ref=${REGISTRY}/myapp:cache,mode=max"]
}

target "app-debug" {
  inherits = ["_common"]
  target   = "debug"
  tags     = ["${REGISTRY}/myapp:debug"]
}
```
