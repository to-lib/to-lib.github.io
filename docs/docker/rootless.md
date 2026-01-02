---
sidebar_position: 29
title: Rootless Docker
description: 无 root 权限运行 Docker，提升安全性
---

# Rootless Docker

Rootless 模式允许以非 root 用户运行 Docker 守护进程和容器，显著提升安全性。

## 为什么使用 Rootless

- **安全性** - 即使容器逃逸，攻击者也只有普通用户权限
- **多租户** - 每个用户可以运行自己的 Docker 实例
- **合规性** - 满足某些安全合规要求

## 前提条件

### 系统要求

```bash
# 检查内核版本（需要 5.11+ 或带补丁的旧版本）
uname -r

# 检查用户命名空间支持
cat /proc/sys/kernel/unprivileged_userns_clone
# 应该输出 1

# 如果输出 0，启用它
echo 1 | sudo tee /proc/sys/kernel/unprivileged_userns_clone
```

### 安装依赖

```bash
# Ubuntu/Debian
sudo apt-get install -y uidmap dbus-user-session

# CentOS/RHEL
sudo yum install -y shadow-utils fuse-overlayfs

# 配置 subuid/subgid
echo "$USER:100000:65536" | sudo tee -a /etc/subuid
echo "$USER:100000:65536" | sudo tee -a /etc/subgid
```

## 安装 Rootless Docker

### 使用官方脚本

```bash
# 如果已安装 root 模式 Docker，先停止
sudo systemctl disable --now docker.service docker.socket

# 安装 rootless
curl -fsSL https://get.docker.com/rootless | sh

# 设置环境变量（添加到 ~/.bashrc）
export PATH=/home/$USER/bin:$PATH
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
```

### 手动安装

```bash
# 下载二进制文件
curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-24.0.7.tgz | tar xz

# 安装 rootless 组件
curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-rootless-extras-24.0.7.tgz | tar xz

# 移动到用户目录
mkdir -p ~/bin
mv docker/* ~/bin/
mv docker-rootless-extras/* ~/bin/
```

## 启动和管理

### 启动 Rootless Docker

```bash
# 手动启动
dockerd-rootless.sh

# 使用 systemd（用户服务）
systemctl --user start docker
systemctl --user enable docker

# 设置开机自启（需要 lingering）
sudo loginctl enable-linger $USER
```

### 验证安装

```bash
# 检查 Docker 信息
docker info

# 查看 Security Options
docker info --format '{{.SecurityOptions}}'
# 应该包含 rootless

# 运行测试容器
docker run hello-world
```

## 配置

### daemon.json

```bash
# 配置文件位置
mkdir -p ~/.config/docker
cat > ~/.config/docker/daemon.json << 'EOF'
{
  "storage-driver": "fuse-overlayfs"
}
EOF
```

### 数据目录

```bash
# 默认数据目录
~/.local/share/docker

# 自定义数据目录
# ~/.config/docker/daemon.json
{
  "data-root": "/path/to/custom/docker"
}
```

## 限制和解决方案

### 端口限制

```bash
# 默认无法绑定 < 1024 端口
# 解决方案 1：使用高端口
docker run -p 8080:80 nginx

# 解决方案 2：设置 sysctl
echo "net.ipv4.ip_unprivileged_port_start=80" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# 解决方案 3：使用 setcap
sudo setcap cap_net_bind_service=ep $(which rootlesskit)
```

### 网络限制

```bash
# Rootless 使用 slirp4netns 或 pasta 进行网络
# 性能略低于 root 模式

# 使用 pasta（性能更好）
# ~/.config/docker/daemon.json
{
  "features": {
    "containerd-snapshotter": true
  }
}
```

### 存储驱动

```bash
# 推荐使用 fuse-overlayfs
# 安装
sudo apt-get install fuse-overlayfs

# 配置
{
  "storage-driver": "fuse-overlayfs"
}
```

### 不支持的功能

| 功能 | 状态 | 替代方案 |
|------|------|----------|
| --privileged | 不支持 | 使用 --cap-add |
| AppArmor | 不支持 | 使用 seccomp |
| Checkpoint | 不支持 | - |
| Overlay network | 有限支持 | 使用 bridge |

## 与 Root 模式共存

```bash
# 可以同时运行 root 和 rootless Docker
# 使用不同的 DOCKER_HOST

# Root 模式
sudo docker ps

# Rootless 模式
docker ps  # 使用用户的 DOCKER_HOST

# 或明确指定
docker -H unix:///run/user/$(id -u)/docker.sock ps
```

## Docker Compose

```bash
# Rootless 模式下正常使用
docker compose up -d

# 确保 DOCKER_HOST 已设置
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
```

## 故障排查

```bash
# 检查 dockerd 日志
journalctl --user -u docker

# 检查 subuid/subgid
cat /etc/subuid
cat /etc/subgid

# 检查用户命名空间
cat /proc/self/uid_map

# 重置 rootless Docker
rm -rf ~/.local/share/docker
systemctl --user restart docker
```
