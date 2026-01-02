---
sidebar_position: 30
title: Docker Init
description: 使用 docker init 自动生成 Dockerfile 和 Compose 配置
---

# Docker Init

`docker init` 是 Docker 官方提供的 CLI 工具，可以自动检测项目类型并生成最佳实践的 Dockerfile 和 docker-compose.yml。

## 基本使用

```bash
# 在项目根目录运行
docker init

# 交互式选择项目类型
? What application platform does your project use?
  Go
  Python
  Node
  Rust
  ASP.NET Core
  PHP with Apache
  Java
  Other
```

## 支持的项目类型

| 类型 | 检测方式 | 生成文件 |
|------|----------|----------|
| Go | go.mod | Dockerfile, compose.yaml |
| Python | requirements.txt, pyproject.toml | Dockerfile, compose.yaml |
| Node | package.json | Dockerfile, compose.yaml |
| Rust | Cargo.toml | Dockerfile, compose.yaml |
| ASP.NET | *.csproj | Dockerfile, compose.yaml |
| PHP | composer.json | Dockerfile, compose.yaml |
| Java | pom.xml, build.gradle | Dockerfile, compose.yaml |

## 生成示例

### Node.js 项目

```bash
$ docker init

Welcome to the Docker Init CLI!

This utility will walk you through creating the following files:
  - .dockerignore
  - Dockerfile
  - compose.yaml

? What application platform does your project use? Node
? What version of Node do you want to use? 18
? Which package manager do you want to use? npm
? What command do you want to use to start the app? npm start
? What port does your server listen on? 3000
```

生成的 Dockerfile：

```dockerfile
# syntax=docker/dockerfile:1

ARG NODE_VERSION=18

FROM node:${NODE_VERSION}-alpine

ENV NODE_ENV production

WORKDIR /usr/src/app

RUN --mount=type=bind,source=package.json,target=package.json \
    --mount=type=bind,source=package-lock.json,target=package-lock.json \
    --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev

USER node

COPY . .

EXPOSE 3000

CMD npm start
```

### Python 项目

```bash
$ docker init

? What application platform does your project use? Python
? What version of Python do you want to use? 3.11
? What port does your server listen on? 8000
? What is the command to run your app? python -m uvicorn main:app --host 0.0.0.0
```

生成的 Dockerfile：

```dockerfile
# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

USER appuser

COPY . .

EXPOSE 8000

CMD python -m uvicorn main:app --host 0.0.0.0
```

### Go 项目

```dockerfile
# syntax=docker/dockerfile:1

ARG GO_VERSION=1.21
FROM golang:${GO_VERSION} AS build
WORKDIR /src

RUN --mount=type=cache,target=/go/pkg/mod/ \
    --mount=type=bind,source=go.sum,target=go.sum \
    --mount=type=bind,source=go.mod,target=go.mod \
    go mod download -x

RUN --mount=type=cache,target=/go/pkg/mod/ \
    --mount=type=bind,target=. \
    CGO_ENABLED=0 go build -o /bin/server .

FROM alpine:latest AS final

RUN --mount=type=cache,target=/var/cache/apk \
    apk --update add ca-certificates tzdata && \
    update-ca-certificates

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser
USER appuser

COPY --from=build /bin/server /bin/

EXPOSE 8080

ENTRYPOINT [ "/bin/server" ]
```

## 生成的 compose.yaml

```yaml
services:
  server:
    build:
      context: .
    ports:
      - 3000:3000
    environment:
      NODE_ENV: production
    develop:
      watch:
        - action: rebuild
          path: .
```

## 生成的 .dockerignore

```plaintext
# Include any files or directories that you don't want to be copied to your
# container here (e.g., local build artifacts, temporary files, etc.).
#
# For more help, visit the .dockerignore file reference guide at
# https://docs.docker.com/go/build-context-dockerignore/

**/.DS_Store
**/.classpath
**/.dockerignore
**/.env
**/.git
**/.gitignore
**/.project
**/.settings
**/.toolstarget
**/.vs
**/.vscode
**/*.*proj.user
**/*.dbmdl
**/*.jfm
**/bin
**/charts
**/docker-compose*
**/compose*
**/Dockerfile*
**/node_modules
**/npm-debug.log
**/obj
**/secrets.dev.yaml
**/values.dev.yaml
LICENSE
README.md
```

## 自定义模板

```bash
# 查看可用选项
docker init --help

# 指定项目类型（跳过交互）
docker init --type node

# 指定输出目录
docker init --output ./docker
```

## 最佳实践特性

docker init 生成的配置包含多项最佳实践：

1. **多阶段构建** - 减小镜像体积
2. **非 root 用户** - 提升安全性
3. **BuildKit 缓存** - 加速构建
4. **.dockerignore** - 优化构建上下文
5. **健康检查** - 容器健康监控
6. **环境变量** - 配置外部化
