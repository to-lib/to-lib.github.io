---
sidebar_position: 4
title: Pod 管理
description: Podman Pod 概念与使用详解
---

# Podman Pod 管理

Pod 是 Podman 的核心特性之一，类似于 Kubernetes 中的 Pod 概念。

## 什么是 Pod？

Pod 是一组共享网络命名空间的容器集合，它们可以通过 localhost 相互通信。

## Pod 基础操作

### 创建 Pod

```bash
# 创建空 Pod
podman pod create --name mypod

# 创建并映射端口
podman pod create --name webpod -p 8080:80

# 带共享资源的 Pod
podman pod create --name datapod \
  --share net,ipc \
  -p 8080:80
```

### 管理 Pod

```bash
# 列出 Pod
podman pod ls

# 查看 Pod 详情
podman pod inspect mypod

# 启动/停止/重启
podman pod start mypod
podman pod stop mypod
podman pod restart mypod

# 删除 Pod
podman pod rm mypod
podman pod rm -f mypod  # 强制删除（包括容器）
```

### 在 Pod 中运行容器

```bash
# 创建 Pod
podman pod create --name webpod -p 8080:80

# 添加 web 服务器
podman run -d --pod webpod --name nginx nginx

# 添加应用
podman run -d --pod webpod --name app myapp

# 容器间通过 localhost 通信
# app 容器可以访问 http://localhost:80
```

## 实用示例

### Web 应用 + 数据库

```bash
# 创建 Pod
podman pod create --name myapp-pod -p 3000:3000

# 运行数据库
podman run -d --pod myapp-pod \
  --name db \
  -e POSTGRES_PASSWORD=secret \
  postgres:15

# 运行应用（通过 localhost:5432 访问数据库）
podman run -d --pod myapp-pod \
  --name app \
  -e DATABASE_URL=postgres://postgres:secret@localhost:5432 \
  myapp
```

### 导出为 Kubernetes YAML

```bash
# 生成 YAML
podman generate kube myapp-pod > myapp-pod.yaml

# 从 YAML 创建
podman play kube myapp-pod.yaml

# 删除
podman play kube --down myapp-pod.yaml
```

## Pod 与 Docker Compose

Podman 支持直接运行 Docker Compose 文件：

```bash
# 使用 podman-compose
pip install podman-compose
podman-compose up -d

# 或使用内置 compose（Podman 3.0+）
podman compose up -d
```

## 查看 Pod 资源

```bash
# 查看 Pod 中的容器
podman pod ps
podman ps --pod

# 查看 Pod 日志
podman logs -f --names mypod-nginx
podman logs -f --names mypod-app

# 查看资源使用
podman pod stats mypod
```
