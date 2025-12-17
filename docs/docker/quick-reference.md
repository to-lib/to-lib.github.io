---
sidebar_position: 9
title: 快速参考
description: Docker 命令与配置快速参考
---

# Docker 快速参考

## 容器命令

### 生命周期管理

| 命令                          | 说明               |
| ----------------------------- | ------------------ |
| `docker run <image>`          | 创建并启动容器     |
| `docker run -d <image>`       | 后台运行容器       |
| `docker run -it <image> bash` | 交互式运行         |
| `docker run --rm <image>`     | 退出后自动删除     |
| `docker start <container>`    | 启动已停止容器     |
| `docker stop <container>`     | 停止容器           |
| `docker restart <container>`  | 重启容器           |
| `docker kill <container>`     | 强制停止容器       |
| `docker rm <container>`       | 删除容器           |
| `docker rm -f <container>`    | 强制删除运行中容器 |

### 容器查看

| 命令                                 | 说明            |
| ------------------------------------ | --------------- |
| `docker ps`                          | 查看运行中容器  |
| `docker ps -a`                       | 查看所有容器    |
| `docker ps -q`                       | 只显示容器 ID   |
| `docker logs <container>`            | 查看日志        |
| `docker logs -f <container>`         | 实时查看日志    |
| `docker logs --tail 100 <container>` | 查看最近 100 行 |
| `docker inspect <container>`         | 查看详细信息    |
| `docker stats`                       | 查看资源使用    |
| `docker top <container>`             | 查看进程        |

### 容器操作

| 命令                                   | 说明           |
| -------------------------------------- | -------------- |
| `docker exec -it <container> bash`     | 进入容器       |
| `docker exec <container> <cmd>`        | 执行命令       |
| `docker cp <src> <container>:<dst>`    | 复制文件到容器 |
| `docker cp <container>:<src> <dst>`    | 从容器复制文件 |
| `docker export <container> > file.tar` | 导出容器       |

## docker run 常用参数

```bash
docker run -d \                    # 后台运行
  --name myapp \                   # 容器名
  -p 8080:80 \                     # 端口映射 (主机:容器)
  -p 127.0.0.1:3000:3000 \         # 绑定到本地
  -v /host/path:/container/path \  # 绑定挂载
  -v myvolume:/data \              # 命名卷
  -e KEY=value \                   # 环境变量
  --env-file .env \                # 从文件加载环境变量
  --network mynet \                # 指定网络
  --restart always \               # 重启策略
  --memory 512m \                  # 内存限制
  --cpus 1.0 \                     # CPU 限制
  --user 1000:1000 \               # 指定用户
  --read-only \                    # 只读文件系统
  --rm \                           # 退出后删除
  nginx:alpine
```

## 镜像命令

| 命令                                         | 说明               |
| -------------------------------------------- | ------------------ |
| `docker images`                              | 列出镜像           |
| `docker pull <image>`                        | 拉取镜像           |
| `docker push <image>`                        | 推送镜像           |
| `docker build -t <tag> .`                    | 构建镜像           |
| `docker build -f Dockerfile.prod -t <tag> .` | 指定 Dockerfile    |
| `docker tag <src> <dst>`                     | 添加标签           |
| `docker rmi <image>`                         | 删除镜像           |
| `docker image prune`                         | 清理悬空镜像       |
| `docker image prune -a`                      | 清理所有未使用镜像 |
| `docker history <image>`                     | 查看镜像历史       |
| `docker inspect <image>`                     | 查看镜像详情       |

## Dockerfile 指令

```dockerfile
# 基础镜像
FROM node:18-alpine

# 标签
LABEL maintainer="dev@example.com"
LABEL version="1.0"

# 构建参数
ARG NODE_ENV=production

# 环境变量
ENV NODE_ENV=${NODE_ENV} \
    PORT=3000

# 工作目录
WORKDIR /app

# 复制文件
COPY package*.json ./
COPY --chown=node:node . .

# 执行命令
RUN npm ci --only=production && \
    npm cache clean --force

# 暴露端口
EXPOSE 3000

# 创建用户并切换
RUN addgroup -S app && adduser -S app -G app
USER app

# 数据卷
VOLUME ["/data"]

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# 入口点
ENTRYPOINT ["node"]

# 默认命令
CMD ["server.js"]
```

## Docker Compose

### 基础配置

```yaml
version: "3.8"

services:
  app:
    build: .
    image: myapp:latest
    container_name: myapp
    ports:
      - "3000:3000"
    volumes:
      - ./src:/app/src
      - node_modules:/app/node_modules
    environment:
      - NODE_ENV=development
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15
    volumes:
      - db-data:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD: secret
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  node_modules:
  db-data:

networks:
  backend:
```

### Compose 命令

| 命令                                   | 说明                 |
| -------------------------------------- | -------------------- |
| `docker compose up`                    | 启动服务             |
| `docker compose up -d`                 | 后台启动             |
| `docker compose up --build`            | 重新构建并启动       |
| `docker compose down`                  | 停止并删除容器       |
| `docker compose down -v`               | 同时删除卷           |
| `docker compose ps`                    | 查看服务状态         |
| `docker compose logs`                  | 查看日志             |
| `docker compose logs -f app`           | 实时查看特定服务日志 |
| `docker compose exec app bash`         | 进入容器             |
| `docker compose run --rm app npm test` | 运行一次性命令       |
| `docker compose build`                 | 构建镜像             |
| `docker compose pull`                  | 拉取镜像             |
| `docker compose restart`               | 重启服务             |

## 网络命令

| 命令                                              | 说明              |
| ------------------------------------------------- | ----------------- |
| `docker network ls`                               | 列出网络          |
| `docker network create <name>`                    | 创建网络          |
| `docker network create --driver overlay <name>`   | 创建 overlay 网络 |
| `docker network inspect <name>`                   | 查看网络详情      |
| `docker network connect <network> <container>`    | 连接容器到网络    |
| `docker network disconnect <network> <container>` | 断开网络连接      |
| `docker network rm <name>`                        | 删除网络          |
| `docker network prune`                            | 清理未使用网络    |

## 卷命令

| 命令                           | 说明         |
| ------------------------------ | ------------ |
| `docker volume ls`             | 列出卷       |
| `docker volume create <name>`  | 创建卷       |
| `docker volume inspect <name>` | 查看卷详情   |
| `docker volume rm <name>`      | 删除卷       |
| `docker volume prune`          | 清理未使用卷 |

## 系统命令

| 命令                               | 说明             |
| ---------------------------------- | ---------------- |
| `docker info`                      | 查看 Docker 信息 |
| `docker version`                   | 查看版本         |
| `docker system df`                 | 查看磁盘使用     |
| `docker system prune`              | 清理未使用资源   |
| `docker system prune -a --volumes` | 彻底清理         |
| `docker events`                    | 查看事件流       |

## 常用端口参考

| 服务          | 默认端口    |
| ------------- | ----------- |
| Nginx         | 80, 443     |
| MySQL         | 3306        |
| PostgreSQL    | 5432        |
| Redis         | 6379        |
| MongoDB       | 27017       |
| Elasticsearch | 9200, 9300  |
| RabbitMQ      | 5672, 15672 |
| Kafka         | 9092        |
| Prometheus    | 9090        |
| Grafana       | 3000        |

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
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container>

# 进入最近创建的容器
docker exec -it $(docker ps -lq) bash

# 导出镜像
docker save myimage:tag | gzip > myimage.tar.gz

# 导入镜像
gunzip -c myimage.tar.gz | docker load

# 查看容器资源使用（格式化）
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## 环境变量配置

### .env 文件

```plaintext
# .env
COMPOSE_PROJECT_NAME=myproject
APP_VERSION=1.0.0
DB_PASSWORD=secret
NODE_ENV=production
```

### 在 Compose 中使用

```yaml
services:
  app:
    image: myapp:${APP_VERSION:-latest}
    environment:
      - NODE_ENV=${NODE_ENV}
      - DB_PASSWORD=${DB_PASSWORD}
```

## 常用镜像

| 镜像                    | 说明       |
| ----------------------- | ---------- |
| `nginx:alpine`          | Web 服务器 |
| `node:18-alpine`        | Node.js    |
| `python:3.11-alpine`    | Python     |
| `openjdk:17-jdk-alpine` | Java       |
| `golang:1.21-alpine`    | Go         |
| `mysql:8.0`             | MySQL      |
| `postgres:15`           | PostgreSQL |
| `redis:alpine`          | Redis      |
| `mongo:6`               | MongoDB    |
