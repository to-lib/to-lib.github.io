---
sidebar_position: 3
title: 基础命令
description: Docker 常用命令详解
---

# Docker 基础命令

本文介绍 Docker 最常用的命令及其使用方法。

## 镜像命令

### 拉取镜像

```bash
# 拉取最新版本
docker pull nginx

# 拉取指定版本
docker pull nginx:1.24

# 拉取指定仓库镜像
docker pull registry.example.com/myimage:tag
```

### 查看镜像

```bash
# 列出本地镜像
docker images

# 显示镜像 ID
docker images -q

# 显示所有镜像（包括中间层）
docker images -a

# 按条件过滤
docker images --filter "dangling=true"
```

### 删除镜像

```bash
# 删除指定镜像
docker rmi nginx

# 强制删除
docker rmi -f nginx

# 删除所有未使用镜像
docker image prune

# 删除所有镜像
docker rmi $(docker images -q)
```

### 镜像信息

```bash
# 查看镜像详情
docker inspect nginx

# 查看镜像历史
docker history nginx

# 搜索镜像
docker search nginx
```

## 容器命令

### 创建和运行容器

```bash
# 运行容器（前台）
docker run nginx

# 后台运行
docker run -d nginx

# 指定容器名称
docker run -d --name my-nginx nginx

# 端口映射
docker run -d -p 8080:80 nginx

# 挂载卷
docker run -d -v /host/path:/container/path nginx

# 设置环境变量
docker run -d -e MYSQL_ROOT_PASSWORD=123456 mysql

# 交互式运行
docker run -it ubuntu bash

# 自动删除（退出后）
docker run --rm -it alpine sh

# 资源限制
docker run -d --memory="512m" --cpus="1.0" nginx
```

### 查看容器

```bash
# 查看运行中的容器
docker ps

# 查看所有容器
docker ps -a

# 只显示容器 ID
docker ps -q

# 显示容器大小
docker ps -s
```

### 容器生命周期

```bash
# 启动已停止的容器
docker start container_name

# 停止容器
docker stop container_name

# 重启容器
docker restart container_name

# 强制停止
docker kill container_name

# 暂停容器
docker pause container_name

# 恢复容器
docker unpause container_name

# 删除容器
docker rm container_name

# 强制删除运行中的容器
docker rm -f container_name

# 删除所有已停止容器
docker container prune
```

### 容器操作

```bash
# 进入运行中的容器
docker exec -it container_name bash

# 执行单个命令
docker exec container_name ls -la

# 查看容器日志
docker logs container_name

# 实时查看日志
docker logs -f container_name

# 查看最近 100 行日志
docker logs --tail 100 container_name

# 查看容器详情
docker inspect container_name

# 查看容器进程
docker top container_name

# 查看容器资源使用
docker stats container_name
```

### 容器与主机交互

```bash
# 从容器复制文件到主机
docker cp container_name:/path/file /host/path

# 从主机复制文件到容器
docker cp /host/path/file container_name:/path

# 导出容器为 tar 文件
docker export container_name > container.tar

# 从 tar 文件导入为镜像
docker import container.tar myimage:tag
```

## 网络命令

```bash
# 列出网络
docker network ls

# 创建网络
docker network create mynetwork

# 创建 bridge 网络
docker network create --driver bridge mybridge

# 查看网络详情
docker network inspect mynetwork

# 连接容器到网络
docker network connect mynetwork container_name

# 断开网络连接
docker network disconnect mynetwork container_name

# 删除网络
docker network rm mynetwork

# 清理未使用网络
docker network prune
```

## 卷命令

```bash
# 列出卷
docker volume ls

# 创建卷
docker volume create myvolume

# 查看卷详情
docker volume inspect myvolume

# 删除卷
docker volume rm myvolume

# 清理未使用卷
docker volume prune
```

## 系统命令

```bash
# 查看 Docker 信息
docker info

# 查看 Docker 版本
docker version

# 查看磁盘使用
docker system df

# 详细磁盘使用
docker system df -v

# 清理系统
docker system prune

# 清理所有（包括未使用镜像和卷）
docker system prune -a --volumes

# 查看事件
docker events

# 登录仓库
docker login

# 退出登录
docker logout
```

## 常用组合命令

```bash
# 停止所有容器
docker stop $(docker ps -q)

# 删除所有容器
docker rm $(docker ps -aq)

# 删除所有镜像
docker rmi $(docker images -q)

# 删除悬空镜像
docker rmi $(docker images -f "dangling=true" -q)

# 查看容器 IP
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_name

# 进入最近创建的容器
docker exec -it $(docker ps -lq) bash
```

## 命令速查表

| 命令                | 说明             |
| ------------------- | ---------------- |
| `docker run`        | 创建并运行容器   |
| `docker start/stop` | 启动/停止容器    |
| `docker ps`         | 列出容器         |
| `docker images`     | 列出镜像         |
| `docker pull/push`  | 拉取/推送镜像    |
| `docker exec`       | 在容器中执行命令 |
| `docker logs`       | 查看容器日志     |
| `docker build`      | 构建镜像         |
| `docker rm/rmi`     | 删除容器/镜像    |
| `docker network`    | 管理网络         |
| `docker volume`     | 管理卷           |
