---
sidebar_position: 7
title: 网络管理
description: Podman 网络配置与管理详解
---

# Podman 网络管理

Podman 提供了灵活的网络管理功能，支持多种网络驱动和配置选项。

## 网络驱动

Podman 支持两种网络后端：

| 驱动         | 说明           | 版本要求          |
| ------------ | -------------- | ----------------- |
| **CNI**      | 传统网络驱动   | Podman < 4.0 默认 |
| **Netavark** | 新一代网络驱动 | Podman 4.0+ 默认  |

```bash
# 查看当前网络后端
podman info | grep networkBackend
```

## 网络类型

### Bridge 网络

默认网络模式，容器通过虚拟网桥连接：

```bash
# 创建 bridge 网络
podman network create mynet

# 指定子网
podman network create --subnet 192.168.100.0/24 mynet

# 使用网络
podman run -d --network mynet --name web nginx
```

### Host 网络

容器直接使用主机网络栈：

```bash
podman run -d --network host nginx
```

### Macvlan 网络

容器直接连接到物理网络：

```bash
# 创建 macvlan 网络
podman network create -d macvlan \
  --subnet 192.168.1.0/24 \
  --gateway 192.168.1.1 \
  -o parent=eth0 \
  macnet

# 使用
podman run -d --network macnet --ip 192.168.1.100 nginx
```

### None 网络

禁用网络：

```bash
podman run --network none alpine ip addr
```

## 网络管理命令

```bash
# 列出网络
podman network ls

# 查看网络详情
podman network inspect mynet

# 连接容器到网络
podman network connect mynet container_name

# 断开网络
podman network disconnect mynet container_name

# 删除网络
podman network rm mynet

# 清理未使用的网络
podman network prune
```

## 容器间通信

### 同一网络内通信

```bash
# 创建网络
podman network create appnet

# 启动服务
podman run -d --network appnet --name db postgres
podman run -d --network appnet --name app myapp

# app 容器可以通过容器名访问
# 例如：postgres://db:5432
```

### 使用 Pod 通信

Pod 内容器共享网络命名空间，通过 localhost 通信：

```bash
podman pod create --name mypod -p 8080:80
podman run -d --pod mypod --name nginx nginx
podman run -d --pod mypod --name app myapp
# app 访问 nginx: http://localhost:80
```

## DNS 配置

```bash
# 自定义 DNS 服务器
podman run --dns 8.8.8.8 --dns 8.8.4.4 nginx

# 添加 DNS 搜索域
podman run --dns-search example.com nginx

# 添加 hosts 条目
podman run --add-host db:192.168.1.100 myapp
```

## 端口映射

```bash
# 映射单个端口
podman run -p 8080:80 nginx

# 映射到特定 IP
podman run -p 127.0.0.1:8080:80 nginx

# 映射端口范围
podman run -p 8000-8010:8000-8010 myapp

# 随机端口映射
podman run -P nginx  # 映射所有 EXPOSE 端口

# 查看端口映射
podman port container_name
```

## Rootless 网络

Rootless 模式使用特殊的网络实现：

### slirp4netns（传统）

```bash
# 默认 rootless 网络
podman run -p 8080:80 nginx
```

### pasta（推荐，性能更好）

```bash
# 使用 pasta（Podman 4.0+）
podman run --network pasta -p 8080:80 nginx
```

### 端口限制

非 root 用户默认不能绑定 1024 以下端口：

```bash
# 方法1：使用高端口
podman run -p 8080:80 nginx

# 方法2：修改系统设置
sudo sysctl -w net.ipv4.ip_unprivileged_port_start=80

# 方法3：使用 pasta 网络
podman run --network pasta -p 80:80 nginx
```

## 网络配置文件

网络配置存储位置：

- **Netavark**: `~/.local/share/containers/storage/networks/` (rootless)
- **CNI**: `/etc/cni/net.d/` (root) 或 `~/.config/cni/net.d/` (rootless)

### 自定义网络配置

```json
{
  "name": "mynet",
  "driver": "bridge",
  "subnets": [
    {
      "subnet": "10.90.0.0/24",
      "gateway": "10.90.0.1"
    }
  ],
  "dns_enabled": true
}
```

## 常见问题

### 容器无法访问外网

```bash
# 检查 DNS
podman run --rm busybox nslookup google.com

# 检查路由
podman run --rm busybox ip route

# 手动指定 DNS
podman run --dns 8.8.8.8 nginx
```

### IPv6 支持

```bash
# 创建支持 IPv6 的网络
podman network create --ipv6 --subnet fd00::/64 mynet6
```

## 最佳实践

1. **生产环境使用自定义网络**，避免使用默认网络
2. **Rootless 模式优先使用 pasta** 获得更好性能
3. **合理规划子网**，避免与主机网络冲突
4. **使用容器名通信**，避免硬编码 IP 地址
5. **定期清理未使用网络** 释放资源
