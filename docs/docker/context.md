---
sidebar_position: 26
title: Docker Context
description: 多环境管理与远程 Docker 主机切换
---

# Docker Context

Docker Context 允许你管理多个 Docker 环境，轻松在本地、远程主机、云服务之间切换。

## 基本概念

Context 存储了连接 Docker 守护进程所需的所有信息：
- 端点地址（本地 socket 或远程 URL）
- TLS 证书配置
- Kubernetes 配置（可选）

## 查看和管理 Context

```bash
# 列出所有 context
docker context ls

# 输出示例
NAME        DESCRIPTION                               DOCKER ENDPOINT
default *   Current DOCKER_HOST based configuration   unix:///var/run/docker.sock
dev         Development server                        ssh://dev@192.168.1.100
prod        Production server                         tcp://prod.example.com:2376

# 查看当前 context
docker context show

# 查看 context 详情
docker context inspect default
```

## 创建 Context

### 本地 Socket

```bash
docker context create local \
  --docker "host=unix:///var/run/docker.sock"
```

### SSH 连接

```bash
# 基本 SSH 连接
docker context create dev \
  --docker "host=ssh://user@192.168.1.100"

# 指定 SSH 密钥
docker context create dev \
  --docker "host=ssh://user@192.168.1.100" \
  --description "Development server"

# SSH 配置会使用 ~/.ssh/config
```

### TCP 连接（带 TLS）

```bash
docker context create prod \
  --docker "host=tcp://prod.example.com:2376,ca=/path/ca.pem,cert=/path/cert.pem,key=/path/key.pem"
```

## 切换 Context

```bash
# 切换默认 context
docker context use dev

# 临时使用其他 context（单次命令）
docker --context prod ps

# 使用环境变量
export DOCKER_CONTEXT=prod
docker ps
```

## 远程 Docker 配置

### 服务端配置（远程主机）

```json
// /etc/docker/daemon.json
{
  "hosts": ["unix:///var/run/docker.sock", "tcp://0.0.0.0:2376"],
  "tls": true,
  "tlscacert": "/etc/docker/certs/ca.pem",
  "tlscert": "/etc/docker/certs/server-cert.pem",
  "tlskey": "/etc/docker/certs/server-key.pem",
  "tlsverify": true
}
```

### 生成 TLS 证书

```bash
# 创建 CA
openssl genrsa -out ca-key.pem 4096
openssl req -new -x509 -days 365 -key ca-key.pem -sha256 -out ca.pem

# 创建服务端证书
openssl genrsa -out server-key.pem 4096
openssl req -subj "/CN=docker-server" -sha256 -new -key server-key.pem -out server.csr
echo "subjectAltName = DNS:docker-server,IP:192.168.1.100" > extfile.cnf
openssl x509 -req -days 365 -sha256 -in server.csr -CA ca.pem -CAkey ca-key.pem \
  -CAcreateserial -out server-cert.pem -extfile extfile.cnf

# 创建客户端证书
openssl genrsa -out key.pem 4096
openssl req -subj '/CN=client' -new -key key.pem -out client.csr
echo "extendedKeyUsage = clientAuth" > extfile-client.cnf
openssl x509 -req -days 365 -sha256 -in client.csr -CA ca.pem -CAkey ca-key.pem \
  -CAcreateserial -out cert.pem -extfile extfile-client.cnf
```

## SSH 方式（推荐）

SSH 方式更简单安全，无需配置 TLS。

### 前提条件

```bash
# 确保远程主机已安装 Docker
# 确保 SSH 密钥认证已配置

# 测试 SSH 连接
ssh user@remote-host docker version
```

### 创建 SSH Context

```bash
docker context create remote --docker "host=ssh://user@remote-host"
docker context use remote

# 现在所有 docker 命令都在远程执行
docker ps
docker images
```

### SSH 配置优化

```bash
# ~/.ssh/config
Host docker-dev
    HostName 192.168.1.100
    User deploy
    IdentityFile ~/.ssh/docker_key
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600

# 使用 SSH 别名
docker context create dev --docker "host=ssh://docker-dev"
```

## 云服务集成

### AWS ECS

```bash
# 安装 ECS 插件后
docker context create ecs myecs --from-env
docker context use myecs

# 使用 Compose 部署到 ECS
docker compose up
```

### Azure ACI

```bash
# 登录 Azure
docker login azure

# 创建 ACI context
docker context create aci myaci

# 部署容器
docker context use myaci
docker run -d -p 80:80 nginx
```

## 实用技巧

### 快速切换脚本

```bash
# ~/.bashrc 或 ~/.zshrc
alias ddev='docker context use dev'
alias dprod='docker context use prod'
alias dlocal='docker context use default'

# 显示当前 context 的 PS1
export PS1='[$(docker context show)] \w $ '
```

### 批量操作多环境

```bash
#!/bin/bash
# deploy-all.sh
for ctx in dev staging prod; do
  echo "Deploying to $ctx..."
  docker --context $ctx compose pull
  docker --context $ctx compose up -d
done
```

### Context 导出导入

```bash
# 导出 context
docker context export prod > prod-context.tar

# 导入 context
docker context import prod-backup prod-context.tar
```

## 删除 Context

```bash
# 删除 context
docker context rm dev

# 无法删除当前使用的 context
docker context use default
docker context rm dev
```
