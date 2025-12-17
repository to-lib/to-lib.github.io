---
sidebar_position: 6
title: 网络配置
description: Docker 网络模式与配置详解
---

# Docker 网络配置

Docker 提供了灵活的网络功能，使容器可以相互通信或与外部网络交互。

## 网络驱动类型

| 驱动        | 说明                                 | 使用场景             |
| ----------- | ------------------------------------ | -------------------- |
| **bridge**  | 默认网络驱动，创建独立的网络命名空间 | 单主机容器通信       |
| **host**    | 容器直接使用主机网络                 | 需要最佳网络性能     |
| **none**    | 禁用网络                             | 完全隔离的容器       |
| **overlay** | 跨主机容器通信                       | Docker Swarm 集群    |
| **macvlan** | 容器拥有独立 MAC 地址                | 需要直接访问物理网络 |

## Bridge 网络

### 默认 Bridge 网络

```bash
# 使用默认 bridge 网络运行容器
docker run -d --name web nginx

# 查看默认网络
docker network inspect bridge
```

### 自定义 Bridge 网络

```bash
# 创建自定义网络
docker network create mynetwork

# 指定子网和网关
docker network create \
  --driver bridge \
  --subnet 172.20.0.0/16 \
  --gateway 172.20.0.1 \
  mynetwork

# 使用自定义网络运行容器
docker run -d --name web --network mynetwork nginx
docker run -d --name db --network mynetwork mysql

# 容器间可通过名称通信
docker exec web ping db
```

:::tip
自定义 bridge 网络提供自动 DNS 解析，容器可以通过名称相互访问。
:::

### 连接多个网络

```bash
# 创建两个网络
docker network create frontend
docker network create backend

# 创建容器并连接到多个网络
docker run -d --name app --network frontend nginx
docker network connect backend app

# 查看容器网络
docker inspect app --format '{{json .NetworkSettings.Networks}}'
```

## Host 网络

```bash
# 使用 host 网络
docker run -d --network host nginx

# 容器直接使用主机端口，无需端口映射
# nginx 直接监听主机的 80 端口
```

:::warning
host 网络模式在 macOS 和 Windows 上的行为与 Linux 不同。
:::

## None 网络

```bash
# 禁用网络
docker run -d --network none alpine sleep 3600

# 容器完全没有网络访问
```

## 端口映射

```bash
# 映射单个端口
docker run -d -p 8080:80 nginx

# 映射多个端口
docker run -d -p 80:80 -p 443:443 nginx

# 指定主机 IP
docker run -d -p 127.0.0.1:8080:80 nginx

# 映射端口范围
docker run -d -p 8080-8090:80-90 myapp

# 随机主机端口
docker run -d -P nginx

# 查看端口映射
docker port container_name
```

## DNS 配置

```bash
# 指定 DNS 服务器
docker run --dns 8.8.8.8 --dns 8.8.4.4 nginx

# 指定 DNS 搜索域
docker run --dns-search example.com nginx

# 添加 hosts 记录
docker run --add-host db:192.168.1.100 nginx
```

### 在 daemon.json 中配置

```json
{
  "dns": ["8.8.8.8", "8.8.4.4"],
  "dns-search": ["example.com"]
}
```

## 网络命令

```bash
# 列出网络
docker network ls

# 创建网络
docker network create [OPTIONS] NETWORK

# 查看网络详情
docker network inspect NETWORK

# 连接容器到网络
docker network connect NETWORK CONTAINER

# 断开网络连接
docker network disconnect NETWORK CONTAINER

# 删除网络
docker network rm NETWORK

# 清理未使用网络
docker network prune
```

## Docker Compose 网络

```yaml
version: "3.8"

services:
  frontend:
    image: nginx
    networks:
      - frontend-net

  backend:
    image: node:18
    networks:
      - frontend-net
      - backend-net

  database:
    image: postgres
    networks:
      - backend-net

networks:
  frontend-net:
    driver: bridge
  backend-net:
    driver: bridge
    internal: true # 仅内部访问
```

### 网络别名

```yaml
services:
  db:
    image: postgres
    networks:
      backend:
        aliases:
          - database
          - postgres

networks:
  backend:
```

### 静态 IP

```yaml
services:
  app:
    image: nginx
    networks:
      mynet:
        ipv4_address: 172.28.0.10

networks:
  mynet:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

## 容器间通信

### 同一网络

```bash
# 创建网络
docker network create app-network

# 运行容器
docker run -d --name redis --network app-network redis
docker run -d --name app --network app-network myapp

# app 容器可以通过 'redis' 名称访问 Redis
# 例如：redis://redis:6379
```

### 跨网络通信

```bash
# 通过连接到同一网络
docker network connect shared-network container1
docker network connect shared-network container2

# 或通过主机网络
docker run --network host myapp
```

## 网络排障

```bash
# 查看容器 IP
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container

# 查看网络详情
docker network inspect mynetwork

# 容器内网络测试
docker exec container ping other-container
docker exec container curl http://other-container:8080

# 查看端口监听
docker exec container netstat -tlnp

# 查看 iptables 规则
sudo iptables -L -n -t nat
```

## 常见网络模式对比

| 模式    | 隔离性 | 性能 | 端口映射 | 使用场景         |
| ------- | ------ | ---- | -------- | ---------------- |
| bridge  | 高     | 中   | 需要     | 默认，大多数场景 |
| host    | 无     | 最高 | 不需要   | 高性能需求       |
| none    | 完全   | -    | 不可用   | 安全敏感应用     |
| overlay | 高     | 中   | 需要     | 多主机集群       |
