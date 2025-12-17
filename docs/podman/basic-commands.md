---
sidebar_position: 3
title: 基础命令
description: Podman 常用命令详解
---

# Podman 基础命令

Podman 命令与 Docker 高度兼容，大多数 Docker 命令可直接替换为 Podman。

## 镜像管理

```bash
# 拉取镜像
podman pull nginx:alpine

# 列出镜像
podman images

# 搜索镜像
podman search nginx

# 删除镜像
podman rmi nginx

# 构建镜像
podman build -t myapp:v1 .

# 推送镜像
podman push myapp:v1 registry.example.com/myapp:v1

# 保存和加载镜像
podman save -o nginx.tar nginx
podman load -i nginx.tar
```

## 容器管理

```bash
# 运行容器
podman run -d --name web -p 8080:80 nginx

# 交互式运行
podman run -it alpine sh

# 列出容器
podman ps        # 运行中
podman ps -a     # 所有

# 停止/启动/重启
podman stop web
podman start web
podman restart web

# 删除容器
podman rm web
podman rm -f web  # 强制

# 查看日志
podman logs web
podman logs -f web  # 实时

# 进入容器
podman exec -it web bash

# 查看容器详情
podman inspect web

# 查看资源使用
podman stats
```

## 与 Docker 差异

### 命令对比

| Docker 命令      | Podman 等效命令                      |
| ---------------- | ------------------------------------ |
| `docker run`     | `podman run`                         |
| `docker build`   | `podman build` 或 `buildah bud`      |
| `docker-compose` | `podman-compose` 或 `podman compose` |

### 独有功能

```bash
# 生成 systemd 服务
podman generate systemd --new --name web > web.service

# 生成 Kubernetes YAML
podman generate kube web > web.yaml

# 从 K8s YAML 运行
podman play kube web.yaml

# 检查点和恢复（实验性）
podman container checkpoint web
podman container restore web
```

## 卷管理

```bash
# 创建卷
podman volume create mydata

# 使用卷
podman run -v mydata:/data nginx

# 列出卷
podman volume ls

# 删除卷
podman volume rm mydata

# 清理未使用卷
podman volume prune
```

## 网络管理

```bash
# 创建网络
podman network create mynet

# 使用网络
podman run --network mynet nginx

# 列出网络
podman network ls

# 删除网络
podman network rm mynet
```

## 系统管理

```bash
# 查看信息
podman info

# 清理系统
podman system prune

# 重置 Podman
podman system reset

# Rootless 迁移
podman system migrate
```
