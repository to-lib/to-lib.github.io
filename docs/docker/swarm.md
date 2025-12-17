---
sidebar_position: 14
title: Docker Swarm
description: Docker Swarm 集群管理与服务编排
---

# Docker Swarm

Docker Swarm 是 Docker 内置的容器编排工具，用于管理分布式容器集群。

## 核心概念

| 概念        | 说明                                      |
| ----------- | ----------------------------------------- |
| **Swarm**   | Docker 集群，由多个 Docker 节点组成       |
| **Node**    | 集群中的 Docker 实例（Manager 或 Worker） |
| **Service** | 定义要运行的任务，如副本数、网络、存储    |
| **Task**    | 服务的最小调度单位，对应一个容器          |
| **Stack**   | 一组相关服务，类似 Docker Compose         |

## 集群架构

```
                    ┌──────────────────────────────────────┐
                    │              Swarm Cluster           │
                    ├──────────────────────────────────────┤
                    │                                      │
    ┌───────────────┼───────────────┬──────────────────────┤
    │               │               │                      │
    ▼               ▼               ▼                      ▼
┌────────┐    ┌────────┐    ┌────────────┐    ┌────────────┐
│Manager │    │Manager │    │   Worker   │    │   Worker   │
│ (Leader)│    │(Replica)│    │            │    │            │
└────────┘    └────────┘    └────────────┘    └────────────┘
```

## 集群初始化

### 创建 Swarm 集群

```bash
# 初始化集群（当前节点成为 Manager）
docker swarm init --advertise-addr <MANAGER_IP>

# 输出示例：
# docker swarm join --token SWMTKN-1-xxx <MANAGER_IP>:2377
```

### 添加节点

```bash
# 获取 Worker 加入令牌
docker swarm join-token worker

# 获取 Manager 加入令牌
docker swarm join-token manager

# 在其他节点执行加入命令
docker swarm join --token <TOKEN> <MANAGER_IP>:2377
```

### 节点管理

```bash
# 查看节点
docker node ls

# 查看节点详情
docker node inspect <NODE_ID>

# 更新节点角色
docker node promote <NODE_ID>   # Worker → Manager
docker node demote <NODE_ID>    # Manager → Worker

# 更新节点标签
docker node update --label-add env=production <NODE_ID>

# 设置节点可用性
docker node update --availability drain <NODE_ID>   # 排空节点
docker node update --availability active <NODE_ID>  # 恢复节点

# 删除节点
docker node rm <NODE_ID>
```

### 离开集群

```bash
# Worker 节点离开
docker swarm leave

# Manager 节点强制离开
docker swarm leave --force
```

## 服务管理

### 创建服务

```bash
# 创建简单服务
docker service create --name nginx nginx:alpine

# 指定副本数
docker service create --name nginx --replicas 3 nginx:alpine

# 端口映射
docker service create --name nginx -p 80:80 nginx:alpine

# 环境变量
docker service create --name app \
  -e NODE_ENV=production \
  -e DB_HOST=db \
  myapp:latest

# 挂载卷
docker service create --name db \
  --mount type=volume,source=db-data,target=/var/lib/mysql \
  mysql:8.0

# 资源限制
docker service create --name app \
  --limit-cpu 1 \
  --limit-memory 512M \
  --reserve-cpu 0.5 \
  --reserve-memory 256M \
  myapp:latest

# 部署约束
docker service create --name app \
  --constraint 'node.role==worker' \
  --constraint 'node.labels.env==production' \
  myapp:latest
```

### 查看服务

```bash
# 列出服务
docker service ls

# 查看服务详情
docker service inspect nginx

# 查看服务任务
docker service ps nginx

# 查看服务日志
docker service logs nginx
docker service logs -f nginx
```

### 更新服务

```bash
# 扩展副本
docker service scale nginx=5

# 批量扩展
docker service scale nginx=5 redis=3

# 更新镜像
docker service update --image nginx:1.25 nginx

# 更新环境变量
docker service update --env-add NEW_VAR=value nginx
docker service update --env-rm OLD_VAR nginx

# 更新资源限制
docker service update --limit-memory 1G nginx
```

### 滚动更新

```bash
# 配置滚动更新策略
docker service update \
  --update-parallelism 2 \      # 同时更新 2 个副本
  --update-delay 10s \          # 每批更新间隔 10 秒
  --update-failure-action rollback \  # 失败时回滚
  --update-max-failure-ratio 0.2 \    # 最大失败率 20%
  --image nginx:1.25 \
  nginx
```

### 回滚服务

```bash
# 回滚到上一版本
docker service rollback nginx

# 查看回滚状态
docker service ps nginx
```

### 删除服务

```bash
docker service rm nginx
```

## 网络

### 创建 Overlay 网络

```bash
# 创建 overlay 网络
docker network create --driver overlay mynetwork

# 加密网络
docker network create --driver overlay --opt encrypted mynetwork

# 使用网络创建服务
docker service create --name app --network mynetwork myapp
```

### Ingress 网络

Swarm 自动创建 ingress 网络用于负载均衡：

```bash
# 发布端口时，流量通过 ingress 网络路由
docker service create --name web -p 80:80 nginx

# 访问任意节点的 80 端口都能访问服务
```

## Stack 部署

### Stack 配置文件

```yaml
# docker-compose.yml (用于 Stack)
version: "3.8"

services:
  web:
    image: nginx:alpine
    ports:
      - "80:80"
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      rollback_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      placement:
        constraints:
          - node.role == worker
      resources:
        limits:
          cpus: "0.5"
          memory: 256M
        reservations:
          cpus: "0.25"
          memory: 128M
    networks:
      - frontend

  api:
    image: myapi:latest
    deploy:
      replicas: 2
    networks:
      - frontend
      - backend
    secrets:
      - db_password

  db:
    image: postgres:15
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.db == true
    volumes:
      - db-data:/var/lib/postgresql/data
    networks:
      - backend
    secrets:
      - db_password

networks:
  frontend:
  backend:
    internal: true

volumes:
  db-data:

secrets:
  db_password:
    external: true
```

### Stack 命令

```bash
# 部署 Stack
docker stack deploy -c docker-compose.yml myapp

# 查看 Stack
docker stack ls

# 查看 Stack 服务
docker stack services myapp

# 查看 Stack 任务
docker stack ps myapp

# 删除 Stack
docker stack rm myapp
```

## Secrets 管理

```bash
# 创建 secret
echo "mypassword" | docker secret create db_password -

# 从文件创建
docker secret create ssl_cert ./cert.pem

# 列出 secrets
docker secret ls

# 查看 secret 详情（不显示内容）
docker secret inspect db_password

# 删除 secret
docker secret rm db_password
```

在服务中使用：

```bash
docker service create --name db \
  --secret db_password \
  -e POSTGRES_PASSWORD_FILE=/run/secrets/db_password \
  postgres:15
```

## Configs 管理

```bash
# 创建 config
docker config create nginx_conf ./nginx.conf

# 在服务中使用
docker service create --name web \
  --config source=nginx_conf,target=/etc/nginx/nginx.conf \
  nginx

# 更新 config
docker config create nginx_conf_v2 ./nginx.conf
docker service update \
  --config-rm nginx_conf \
  --config-add source=nginx_conf_v2,target=/etc/nginx/nginx.conf \
  web
```

## 健康检查

```yaml
services:
  app:
    image: myapp
    deploy:
      replicas: 3
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## 集群维护

### 备份集群状态

```bash
# 在 Manager 节点上
sudo cp -r /var/lib/docker/swarm /backup/swarm-$(date +%Y%m%d)
```

### 恢复集群

```bash
# 停止 Docker
sudo systemctl stop docker

# 恢复备份
sudo rm -rf /var/lib/docker/swarm
sudo cp -r /backup/swarm-20240101 /var/lib/docker/swarm

# 重启并强制新建集群
docker swarm init --force-new-cluster
```

### 更新节点

```bash
# 1. 排空节点
docker node update --availability drain node1

# 2. 更新 Docker
# ...

# 3. 重新激活节点
docker node update --availability active node1
```

## Swarm vs Kubernetes

| 特性     | Docker Swarm   | Kubernetes       |
| -------- | -------------- | ---------------- |
| 复杂度   | 简单，易于上手 | 复杂，学习曲线陡 |
| 安装     | Docker 内置    | 需要单独安装     |
| 扩展性   | 中小规模       | 大规模           |
| 社区     | 较小           | 庞大活跃         |
| 功能     | 基础编排       | 功能丰富         |
| 适用场景 | 简单编排需求   | 企业级复杂场景   |

**建议**：

- 小团队/简单需求 → Docker Swarm
- 大规模/复杂需求 → Kubernetes

## 常用命令速查

```bash
# 集群
docker swarm init
docker swarm join
docker swarm leave
docker node ls

# 服务
docker service create
docker service ls
docker service ps <service>
docker service logs <service>
docker service scale <service>=N
docker service update <service>
docker service rm <service>

# Stack
docker stack deploy -c compose.yml <stack>
docker stack ls
docker stack services <stack>
docker stack rm <stack>

# Secrets
docker secret create
docker secret ls
docker secret rm
```
