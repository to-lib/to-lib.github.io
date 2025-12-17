---
sidebar_position: 13
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

# 重置虚拟机
podman machine rm
podman machine init
```

### 版本过旧

```bash
# 添加官方源获取最新版本
# Ubuntu
. /etc/os-release
sudo sh -c "echo 'deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/ /' > /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list"
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
sudo sysctl -w net.ipv4.ip_unprivileged_port_start=80

# 或使用 pasta 网络 (Podman 4.0+)
podman run --network pasta -p 80:80 nginx
```

### 存储空间权限错误

```bash
# 重置存储
podman system reset

# 检查存储配置
cat ~/.config/containers/storage.conf
```

## 网络问题

### 容器无法访问网络

```bash
# 检查 DNS
podman run --rm busybox nslookup google.com

# 设置 DNS
podman run --dns 8.8.8.8 nginx

# 检查网络模式
podman info | grep network
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

### 主机无法访问容器端口

```bash
# 检查端口映射
podman port container_name

# 检查防火墙
sudo firewall-cmd --list-ports
sudo firewall-cmd --add-port=8080/tcp
```

## 存储问题

### 存储空间不足

```bash
# 清理
podman system prune -a --volumes

# 查看空间使用
podman system df
```

### 挂载失败（Permission denied）

```bash
# Rootless 模式检查权限
podman unshare chown -R 1000:1000 /path/to/data

# SELinux 标签
podman run -v /data:/data:Z nginx  # 私有
podman run -v /data:/data:z nginx  # 共享
```

### 卷数据丢失

```bash
# 使用命名卷而非匿名卷
podman volume create mydata
podman run -v mydata:/data nginx

# 检查卷位置
podman volume inspect mydata
```

## 镜像构建问题

### 构建缓存无效

```bash
# 强制无缓存构建
podman build --no-cache -t myapp .

# 拉取最新基础镜像
podman build --pull -t myapp .
```

### 多阶段构建失败

```bash
# 检查 stage 名称
podman build --target builder -t myapp:builder .
```

### 推送镜像失败

```bash
# 检查登录状态
podman login registry.example.com

# 检查镜像标签
podman tag myapp registry.example.com/myapp
podman push registry.example.com/myapp
```

## Docker 兼容问题

### Compose 不工作

```bash
# 安装 podman-compose
pip install podman-compose

# 或使用内置
podman compose up -d

# 检查版本
podman compose version
```

### Socket 连接失败

```bash
# 启用 socket
systemctl --user enable --now podman.socket
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock

# 验证
curl --unix-socket $XDG_RUNTIME_DIR/podman/podman.sock http://localhost/_ping
```

### 某些 Docker 命令不支持

Podman 不支持以下 Docker 功能：

- Docker Swarm（使用 Kubernetes 替代）
- 某些高级网络选项
- 部分实验性功能

## Systemd 集成问题

### 用户服务不启动

```bash
# 启用 linger
loginctl enable-linger $USER

# 检查服务状态
systemctl --user status container-name
journalctl --user -u container-name
```

### 服务启动容器失败

```bash
# 使用 --new 选项生成
podman generate systemd --new --name container > service.service

# 重载配置
systemctl --user daemon-reload
```

## 性能问题

### Rootless 性能较慢

```bash
# 使用 pasta 网络（更快）
podman run --network pasta nginx

# 使用 fuse-overlayfs
# 在 storage.conf 中配置
[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs"
```

### 容器启动慢

```bash
# 预先拉取镜像
podman pull nginx

# 使用本地镜像缓存
# 检查是否有不必要的启动命令
```

## 安全问题

### 如何以非 root 用户运行应用

```dockerfile
FROM node:18-alpine
RUN addgroup -g 1001 app && adduser -u 1001 -G app -S app
USER app
CMD ["node", "server.js"]
```

### 容器需要特权操作

```bash
# 添加特定 capability（而非 --privileged）
podman run --cap-add=SYS_PTRACE alpine
podman run --cap-add=NET_ADMIN alpine
```

## 其他问题

### 容器启动后立即退出

```bash
# 查看日志
podman logs container_name

# 交互式调试
podman run -it image_name sh

# 检查入口命令
podman inspect image_name | grep -A5 Cmd
```

### 镜像拉取失败

配置镜像源 `~/.config/containers/registries.conf`：

```toml
[registries.search]
registries = ['docker.io', 'quay.io']

[[registry]]
location = "docker.io"
[[registry.mirror]]
location = "mirror.gcr.io"
```

### "Error: OCI runtime error" 错误

```bash
# 重置 Podman
podman system reset

# 检查 runtime
podman info | grep runtime

# 重新安装
sudo dnf reinstall podman
```
