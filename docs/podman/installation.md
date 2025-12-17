---
sidebar_position: 2
title: 安装配置
description: Podman 各平台安装与配置指南
---

# Podman 安装配置

## Linux 安装

### Fedora/RHEL/CentOS

```bash
# Fedora
sudo dnf install podman

# RHEL 8/CentOS 8
sudo dnf module install container-tools

# RHEL 7/CentOS 7
sudo yum install podman
```

### Ubuntu/Debian

```bash
# Ubuntu 22.04+
sudo apt-get update
sudo apt-get install podman

# 添加官方源（旧版本）
. /etc/os-release
echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/ /" | \
  sudo tee /etc/apt/sources.list.d/podman.list
curl -L "https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/Release.key" | \
  sudo apt-key add -
sudo apt-get update
sudo apt-get install podman
```

### Arch Linux

```bash
sudo pacman -S podman
```

## macOS 安装

```bash
# Homebrew
brew install podman

# 初始化机器（需要虚拟机）
podman machine init
podman machine start
```

## Windows 安装

```powershell
# 使用 winget
winget install -e --id RedHat.Podman

# 初始化机器
podman machine init
podman machine start
```

## 配置

### 配置文件位置

- 系统配置：`/etc/containers/`
- 用户配置：`~/.config/containers/`

### 镜像仓库配置

编辑 `~/.config/containers/registries.conf`：

```toml
[registries.search]
registries = ['docker.io', 'quay.io', 'registry.fedoraproject.org']

[[registry]]
location = "docker.io"
[[registry.mirror]]
location = "mirror.gcr.io"
```

### Rootless 配置

```bash
# 设置 subuid/subgid
sudo usermod --add-subuids 100000-165535 $USER
sudo usermod --add-subgids 100000-165535 $USER

# 迁移存储
podman system migrate
```

## 验证安装

```bash
# 查看版本
podman version

# 查看信息
podman info

# 运行测试容器
podman run hello-world
```

## Docker 兼容

```bash
# 创建别名
alias docker=podman

# 或使用 podman-docker 包
sudo apt install podman-docker  # Debian/Ubuntu
sudo dnf install podman-docker  # Fedora
```
