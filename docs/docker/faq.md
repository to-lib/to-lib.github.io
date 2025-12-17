---
sidebar_position: 10
title: 常见问题
description: Docker 常见问题与解决方案
---

# Docker 常见问题

## 安装与配置

### Docker 服务无法启动

```bash
# 查看状态
sudo systemctl status docker

# 查看日志
sudo journalctl -xu docker
```

### 权限不足

```bash
# 添加用户到 docker 组
sudo usermod -aG docker $USER
newgrp docker
```

### 镜像拉取慢

配置镜像加速器 `/etc/docker/daemon.json`：

```json
{
  "registry-mirrors": ["https://mirror.ccs.tencentyun.com"]
}
```

## 容器问题

### 容器启动后立即退出

```bash
# 查看日志
docker logs container_name

# 交互式调试
docker run -it image_name sh
```

### 无法删除容器

```bash
# 强制删除
docker rm -f container_name

# 删除所有
docker rm -f $(docker ps -aq)
```

### 容器无法连接网络

```bash
# 检查 DNS
docker run --rm busybox nslookup google.com

# 使用自定义 DNS
docker run --dns 8.8.8.8 myimage
```

## 镜像问题

### 镜像过大

- 使用 Alpine 基础镜像
- 多阶段构建
- 清理缓存：`rm -rf /var/lib/apt/lists/*`

### 构建失败

```bash
# 查看详细日志
docker build --progress=plain .

# 不使用缓存
docker build --no-cache .
```

## 存储问题

### 磁盘空间不足

```bash
# 清理
docker system prune -a --volumes

# 查看使用情况
docker system df
```

### 数据丢失

使用命名卷而非匿名卷：

```bash
docker run -v mydata:/data nginx
```

## 网络问题

### 端口冲突

```bash
# 查看端口占用
lsof -i :80

# 使用其他端口
docker run -p 8080:80 nginx
```

### 容器间无法通信

```bash
# 使用同一网络
docker network create mynet
docker run --network mynet --name app1 nginx
docker run --network mynet --name app2 alpine ping app1
```
