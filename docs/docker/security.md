---
sidebar_position: 12
title: Docker 安全
description: Docker 安全最佳实践与配置指南
---

# Docker 安全

本文介绍 Docker 容器化环境中的安全最佳实践。

## 镜像安全

### 使用官方镜像

```dockerfile
# ✅ 推荐：使用官方镜像
FROM node:18-alpine
FROM nginx:1.25

# ❌ 避免：使用来源不明的镜像
FROM randomuser/unknown-image
```

### 指定精确版本

```dockerfile
# ✅ 推荐：指定精确版本
FROM node:18.19.0-alpine3.19

# ❌ 避免：使用 latest 标签
FROM node:latest
```

### 镜像扫描

```bash
# 使用 Docker Scout 扫描
docker scout cves myimage:tag

# 使用 Trivy 扫描
trivy image myimage:tag

# 使用 Grype 扫描
grype myimage:tag
```

### 最小化镜像

```dockerfile
# 使用 Alpine
FROM node:18-alpine

# 使用 Distroless（更安全）
FROM gcr.io/distroless/nodejs18

# 使用 scratch（最小化）
FROM scratch
COPY myapp /
```

## 运行时安全

### 使用非 root 用户

```dockerfile
# 创建专用用户
RUN groupadd -r app && useradd -r -g app app

# 设置目录权限
RUN chown -R app:app /app

# 切换用户
USER app
```

验证：

```bash
docker run myimage whoami
# 应输出 app 而不是 root
```

### 只读文件系统

```bash
# 只读根文件系统
docker run --read-only nginx

# 允许写入特定目录
docker run --read-only \
  --tmpfs /tmp \
  --tmpfs /var/run \
  -v logs:/var/log \
  nginx
```

### 资源限制

```bash
docker run -d \
  --memory=512m \
  --memory-swap=512m \    # 禁用 swap
  --cpus=1.0 \
  --pids-limit=100 \       # 防止 fork 炸弹
  --ulimit nofile=1024:1024 \
  nginx
```

### 禁用 capabilities

```bash
# 删除所有 capabilities
docker run --cap-drop=ALL nginx

# 只添加必需的
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE nginx
```

常见 capabilities：

| Capability       | 说明               | 是否必需 |
| ---------------- | ------------------ | -------- |
| NET_BIND_SERVICE | 绑定小于 1024 端口 | 通常需要 |
| CHOWN            | 更改文件所有者     | 很少需要 |
| DAC_OVERRIDE     | 绕过权限检查       | 避免使用 |
| SETUID/SETGID    | 更改用户/组 ID     | 避免使用 |

### 禁用特权模式

```bash
# ❌ 绝对避免在生产环境使用
docker run --privileged nginx

# ✅ 正确做法：仅授予必需权限
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE nginx
```

### 安全选项

```bash
docker run \
  --security-opt=no-new-privileges \  # 禁止提权
  --security-opt=seccomp=default \    # 系统调用过滤
  --security-opt=apparmor=docker-default \  # AppArmor
  nginx
```

## 网络安全

### 隔离敏感网络

```yaml
# docker-compose.yml
services:
  frontend:
    networks:
      - frontend

  backend:
    networks:
      - frontend
      - backend

  database:
    networks:
      - backend

networks:
  frontend:
  backend:
    internal: true # 无法访问外部网络
```

### 限制容器间通信

```bash
# 禁用默认 bridge 网络的 ICC
# /etc/docker/daemon.json
{
  "icc": false
}
```

### 使用 TLS

```bash
# 配置 Docker 守护进程 TLS
dockerd \
  --tlsverify \
  --tlscacert=/certs/ca.pem \
  --tlscert=/certs/server-cert.pem \
  --tlskey=/certs/server-key.pem \
  -H=0.0.0.0:2376
```

## 敏感数据管理

### Docker Secrets（Swarm 模式）

```bash
# 创建 secret
echo "mypassword" | docker secret create db_password -

# 使用 secret
docker service create \
  --name myapp \
  --secret db_password \
  myimage
```

容器内访问：

```bash
cat /run/secrets/db_password
```

### 环境变量安全

```yaml
# ❌ 不安全：明文密码
services:
  db:
    environment:
      MYSQL_PASSWORD: secret123

# ✅ 使用 secrets
services:
  db:
    secrets:
      - db_password
    environment:
      MYSQL_PASSWORD_FILE: /run/secrets/db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

### 构建时敏感信息

```dockerfile
# ❌ 错误：密码会保留在镜像层
RUN echo "password" > /secret && rm /secret

# ✅ 正确：使用 BuildKit secrets
# syntax=docker/dockerfile:1.4
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc npm ci
```

构建命令：

```bash
DOCKER_BUILDKIT=1 docker build --secret id=npmrc,src=.npmrc .
```

## 镜像签名与验证

### Docker Content Trust

```bash
# 启用内容信任
export DOCKER_CONTENT_TRUST=1

# 签名并推送
docker push myregistry/myimage:v1

# 拉取时自动验证
docker pull myregistry/myimage:v1
```

### 配置信任策略

```json
// /etc/docker/daemon.json
{
  "content-trust": {
    "trust-pinning": {
      "official-library-images": true
    }
  }
}
```

## 审计与监控

### 启用审计日志

```json
// /etc/docker/daemon.json
{
  "log-level": "info",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

### 监控系统调用

```bash
# 使用 Falco 监控容器安全事件
docker run -d \
  --name falco \
  --privileged \
  -v /var/run/docker.sock:/var/run/docker.sock \
  falcosecurity/falco
```

## 安全检查清单

### Dockerfile 检查

| 检查项             | 状态 |
| ------------------ | ---- |
| 使用官方基础镜像   | ☐    |
| 指定精确版本       | ☐    |
| 使用非 root 用户   | ☐    |
| 使用多阶段构建     | ☐    |
| 清理不必要文件     | ☐    |
| 使用 .dockerignore | ☐    |

### 运行时检查

| 检查项                  | 状态 |
| ----------------------- | ---- |
| 资源限制已设置          | ☐    |
| 只读文件系统            | ☐    |
| 删除不必要 capabilities | ☐    |
| 禁用特权模式            | ☐    |
| 设置安全选项            | ☐    |

### 网络检查

| 检查项                   | 状态 |
| ------------------------ | ---- |
| 使用自定义网络           | ☐    |
| 敏感服务使用内部网络     | ☐    |
| 仅暴露必要端口           | ☐    |
| 启用 TLS（如需远程访问） | ☐    |

## 安全配置示例

```yaml
# docker-compose.yml
version: "3.8"

services:
  app:
    image: myapp:v1.0.0
    read_only: true
    user: "1000:1000"
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
          pids: 100
    tmpfs:
      - /tmp
    networks:
      - frontend
    secrets:
      - app_secret

  db:
    image: postgres:15
    read_only: true
    user: "999:999"
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    volumes:
      - db-data:/var/lib/postgresql/data
    networks:
      - backend
    secrets:
      - db_password

networks:
  frontend:
  backend:
    internal: true

secrets:
  app_secret:
    file: ./secrets/app.txt
  db_password:
    file: ./secrets/db.txt

volumes:
  db-data:
```
