---
sidebar_position: 9
title: 常见问题
description: Podman 常见问题与解决方案
---

# Podman 常见问题

## 安装问题

### 找不到 podman 命令

```bash
# Fedora/RHEL
sudo dnf install podman

# Ubuntu
sudo apt install podman
```

### macOS/Windows 启动失败

```bash
# 初始化虚拟机
podman machine init
podman machine start
```

## Rootless 问题

### 没有权限映射

```bash
# 检查 subuid/subgid
cat /etc/subuid | grep $USER

# 添加映射
sudo usermod --add-subuids 100000-165535 $USER
sudo usermod --add-subgids 100000-165535 $USER
podman system migrate
```

### 无法绑定低端口

```bash
# 使用高端口
podman run -p 8080:80 nginx

# 或修改系统设置
sudo sysctl net.ipv4.ip_unprivileged_port_start=80
```

## 网络问题

### 容器无法访问网络

```bash
# 检查 DNS
podman run --rm busybox nslookup google.com

# 设置 DNS
podman run --dns 8.8.8.8 nginx
```

### 容器间无法通信

```bash
# 使用同一网络
podman network create mynet
podman run --network mynet --name app1 nginx
podman run --network mynet --name app2 alpine ping app1

# 或使用 Pod
podman pod create --name mypod
podman run --pod mypod --name app1 nginx
podman run --pod mypod --name app2 alpine wget -qO- localhost
```

## 存储问题

### 存储空间不足

```bash
# 清理
podman system prune -a --volumes
```

### 挂载失败

```bash
# Rootless 模式检查权限
podman unshare chown -R 1000:1000 /path/to/data
```

## Docker 兼容问题

### Compose 不工作

```bash
# 安装 podman-compose
pip install podman-compose

# 或使用内置
podman compose up -d
```

### Socket 连接失败

```bash
# 启用 socket
systemctl --user enable --now podman.socket
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock
```

## 其他问题

### 容器启动后退出

```bash
# 查看日志
podman logs container_name

# 交互式调试
podman run -it image_name sh
```

### 镜像拉取失败

配置镜像源 `~/.config/containers/registries.conf`：

```toml
[registries.search]
registries = ['docker.io', 'quay.io']
```
