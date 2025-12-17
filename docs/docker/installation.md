---
sidebar_position: 2
title: 安装配置
description: Docker 各平台安装与配置指南
---

# Docker 安装配置

本文介绍如何在不同操作系统上安装和配置 Docker。

## Linux 安装

### Ubuntu/Debian

```bash
# 更新包索引
sudo apt-get update

# 安装必要依赖
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 添加 Docker 官方 GPG 密钥
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 设置仓库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装 Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

### CentOS/RHEL

```bash
# 安装必要工具
sudo yum install -y yum-utils

# 添加 Docker 仓库
sudo yum-config-manager --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo

# 安装 Docker Engine
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 启动 Docker
sudo systemctl start docker
sudo systemctl enable docker
```

## macOS 安装

### 使用 Docker Desktop

1. 下载 [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. 双击 `.dmg` 文件并拖拽到应用程序文件夹
3. 启动 Docker Desktop

### 使用 Homebrew

```bash
brew install --cask docker
```

## Windows 安装

### 使用 Docker Desktop

1. 确保已启用 WSL 2 或 Hyper-V
2. 下载 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
3. 运行安装程序并按提示完成安装
4. 重启计算机（如需要）

### 启用 WSL 2 后端

```powershell
# 以管理员身份运行 PowerShell
wsl --install
wsl --set-default-version 2
```

## 安装后配置

### 非 root 用户运行 Docker（Linux）

```bash
# 创建 docker 用户组
sudo groupadd docker

# 将当前用户添加到 docker 组
sudo usermod -aG docker $USER

# 重新登录或执行以下命令使更改生效
newgrp docker

# 验证
docker run hello-world
```

### 配置镜像加速

创建或编辑 `/etc/docker/daemon.json`：

```json
{
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://registry.docker-cn.com"
  ]
}
```

重启 Docker 服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 配置存储驱动

```json
{
  "storage-driver": "overlay2",
  "storage-opts": ["overlay2.override_kernel_check=true"]
}
```

### 配置日志驱动

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

## 验证安装

```bash
# 查看 Docker 版本
docker version

# 查看 Docker 信息
docker info

# 运行测试容器
docker run hello-world

# 运行交互式容器
docker run -it ubuntu bash
```

## 常见问题

### Docker 服务无法启动

```bash
# 查看服务状态
sudo systemctl status docker

# 查看日志
sudo journalctl -xu docker.service
```

### 权限问题

```bash
# 检查 docker.sock 权限
ls -la /var/run/docker.sock

# 如果权限不对，修复
sudo chmod 666 /var/run/docker.sock
```

### 网络问题

```bash
# 检查 Docker 网络
docker network ls

# 重置 Docker 网络
docker network prune
```
