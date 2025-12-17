---
sidebar_position: 5
title: Rootless 模式
description: Podman Rootless 容器运行详解
---

# Podman Rootless 模式

Rootless 模式是 Podman 的核心优势，允许非 root 用户运行容器，提高安全性。

## 配置 Rootless

### 设置子 UID/GID

```bash
# 查看当前设置
cat /etc/subuid
cat /etc/subgid

# 添加映射范围
sudo usermod --add-subuids 100000-165535 $USER
sudo usermod --add-subgids 100000-165535 $USER

# 验证
podman unshare cat /proc/self/uid_map
```

### 迁移存储

```bash
# 从 root 模式迁移
podman system migrate
```

## Rootless 运行

```bash
# 直接以普通用户运行
podman run -d -p 8080:80 nginx

# 查看用户命名空间
podman unshare id
```

## 端口限制

非 root 用户默认不能绑定 1024 以下端口：

```bash
# 方法1：使用高端口
podman run -p 8080:80 nginx

# 方法2：修改系统设置
sudo sysctl net.ipv4.ip_unprivileged_port_start=80

# 方法3：端口转发
sudo firewall-cmd --add-forward-port=port=80:proto=tcp:toport=8080
```

## 存储配置

Rootless 存储位于用户目录：

```bash
# 默认位置
~/.local/share/containers/

# 配置文件
~/.config/containers/storage.conf
```

```toml
[storage]
driver = "overlay"

[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs"
```

## 网络配置

Rootless 使用 slirp4netns：

```bash
# 查看网络模式
podman info | grep network

# 使用 pasta（更高性能）
podman run --network pasta nginx
```

## 与 Root 模式对比

| 特性     | Rootless | Root     |
| -------- | -------- | -------- |
| 安全性   | 更高     | 较低     |
| 端口     | >1024    | 所有     |
| 性能     | 略低     | 更高     |
| 存储位置 | 用户目录 | 系统目录 |

## 最佳实践

1. **生产环境优先使用 Rootless**
2. **配置足够的 subuid/subgid 范围**
3. **使用 overlay 存储驱动**
4. **考虑使用 pasta 网络提升性能**
