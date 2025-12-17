---
sidebar_position: 8
title: 快速参考
description: Podman 命令快速参考
---

# Podman 快速参考

## 常用命令

| 命令                           | 说明          |
| ------------------------------ | ------------- |
| `podman run -d -p 80:80 nginx` | 运行容器      |
| `podman ps -a`                 | 列出所有容器  |
| `podman images`                | 列出镜像      |
| `podman exec -it <id> bash`    | 进入容器      |
| `podman logs -f <id>`          | 查看日志      |
| `podman stop/rm <id>`          | 停止/删除容器 |

## 容器运行

```bash
podman run -d \
  --name myapp \
  -p 8080:80 \
  -v /data:/data \
  -e KEY=value \
  --network mynet \
  nginx
```

## Pod 管理

```bash
# 创建 Pod
podman pod create --name mypod -p 8080:80

# 在 Pod 中运行
podman run -d --pod mypod nginx

# 列出 Pod
podman pod ls
```

## Systemd 集成

```bash
# 生成服务
podman generate systemd --new --name web > web.service

# 启用服务
systemctl --user enable --now web
```

## Rootless 配置

```bash
# 配置 subuid/subgid
sudo usermod --add-subuids 100000-165535 $USER
sudo usermod --add-subgids 100000-165535 $USER
```

## 系统清理

```bash
podman system prune -a  # 清理所有
podman volume prune     # 清理卷
podman pod prune        # 清理 Pod
```

## Docker 兼容

```bash
alias docker=podman
# 或
sudo dnf install podman-docker
```
