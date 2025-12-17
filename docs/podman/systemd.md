---
sidebar_position: 6
title: Systemd 集成
description: Podman 与 Systemd 集成使用
---

# Podman Systemd 集成

Podman 可以生成 systemd 服务单元文件，实现容器的系统级管理。

## 生成服务文件

### 从运行中的容器生成

```bash
# 运行容器
podman run -d --name web -p 8080:80 nginx

# 生成 systemd 服务
podman generate systemd --name web > ~/.config/systemd/user/web.service

# 使用 --new 选项（推荐）
podman generate systemd --new --name web > web.service
```

### --new 选项说明

- **不使用 --new**：服务启动/停止现有容器
- **使用 --new**：每次启动时创建新容器，停止时删除

## 用户服务

### 安装服务

```bash
# 创建目录
mkdir -p ~/.config/systemd/user/

# 复制服务文件
cp web.service ~/.config/systemd/user/

# 重载配置
systemctl --user daemon-reload

# 启动服务
systemctl --user start web

# 开机自启
systemctl --user enable web

# 允许用户服务在登出后继续运行
loginctl enable-linger $USER
```

### 管理服务

```bash
# 查看状态
systemctl --user status web

# 查看日志
journalctl --user -u web

# 停止服务
systemctl --user stop web

# 禁用服务
systemctl --user disable web
```

## 系统服务（Root）

```bash
# 生成服务文件
sudo podman generate systemd --new --name web > /etc/systemd/system/web.service

# 管理服务
sudo systemctl daemon-reload
sudo systemctl enable --now web
sudo systemctl status web
```

## 服务文件示例

```ini
# web.service
[Unit]
Description=Podman web container
After=network.target

[Service]
Type=forking
Restart=always
ExecStartPre=-/usr/bin/podman rm -f web
ExecStart=/usr/bin/podman run -d --name web -p 8080:80 nginx
ExecStop=/usr/bin/podman stop web

[Install]
WantedBy=default.target
```

## Pod 服务

```bash
# 创建 Pod
podman pod create --name mypod -p 8080:80
podman run -d --pod mypod --name web nginx

# 生成 Pod 服务
podman generate systemd --new --name mypod > mypod.service
```

## Quadlet（Podman 4.4+）

Quadlet 使用声明式配置管理容器：

```ini
# ~/.config/containers/systemd/web.container
[Container]
Image=nginx
PublishPort=8080:80

[Service]
Restart=always

[Install]
WantedBy=default.target
```

```bash
# 重载并启动
systemctl --user daemon-reload
systemctl --user start web
```

## 最佳实践

1. **优先使用 --new 选项**
2. **生产环境使用 Quadlet**
3. **设置 Restart=always 确保容器自动重启**
4. **使用 loginctl enable-linger 允许用户服务持续运行**
