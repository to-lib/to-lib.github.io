---
sidebar_position: 7
title: Docker 迁移
description: 从 Docker 迁移到 Podman 指南
---

# 从 Docker 迁移到 Podman

本文介绍如何从 Docker 平滑迁移到 Podman。

## 命令兼容性

大多数 Docker 命令可直接替换：

```bash
# 创建别名
alias docker=podman
```

或安装兼容包：

```bash
# Fedora/RHEL
sudo dnf install podman-docker

# Ubuntu/Debian
sudo apt install podman-docker
```

## 迁移步骤

### 1. 导出 Docker 镜像

```bash
# 列出镜像
docker images

# 导出镜像
docker save myimage:tag -o myimage.tar
```

### 2. 导入到 Podman

```bash
podman load -i myimage.tar
```

### 3. 迁移 Docker Compose

```bash
# 安装 podman-compose
pip install podman-compose

# 运行 Compose 文件
podman-compose up -d

# 或使用内置 compose
podman compose up -d
```

## 差异对比

| 功能     | Docker                 | Podman        |
| -------- | ---------------------- | ------------- |
| 守护进程 | 需要                   | 不需要        |
| 默认用户 | root                   | 支持 rootless |
| Socket   | `/var/run/docker.sock` | 无（可模拟）  |
| Swarm    | 支持                   | 不支持        |
| 网络     | bridge/overlay         | CNI 插件      |

## Docker Socket 兼容

```bash
# 启用 Podman socket（用户级）
systemctl --user enable --now podman.socket

# 设置环境变量
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock

# 系统级
sudo systemctl enable --now podman.socket
export DOCKER_HOST=unix:///run/podman/podman.sock
```

## Compose 迁移

### docker-compose.yml 兼容性

大部分语法直接兼容：

```yaml
version: "3.8"
services:
  web:
    image: nginx
    ports:
      - "80:80"
```

### 不支持的功能

- `network_mode: bridge`（Podman 使用不同的网络驱动）
- 某些高级网络选项

## 常见问题

### 网络不同

```bash
# 创建兼容网络
podman network create --driver bridge mynet
```

### 存储位置不同

- Docker: `/var/lib/docker`
- Podman (root): `/var/lib/containers`
- Podman (rootless): `~/.local/share/containers`

### 服务管理

Docker 使用守护进程，Podman 推荐使用 systemd：

```bash
podman generate systemd --new --name web > web.service
```

## 验证迁移

```bash
# 验证镜像
podman images

# 验证运行
podman run --rm nginx echo "Hello from Podman"

# 测试 Docker 兼容
docker version  # 应该显示 Podman 信息
```
