---
sidebar_position: 9
title: 快速参考
description: Docker 命令与配置快速参考
---

# Docker 快速参考

## 常用命令

| 命令                           | 说明          |
| ------------------------------ | ------------- |
| `docker run -d -p 80:80 nginx` | 后台运行容器  |
| `docker ps -a`                 | 列出所有容器  |
| `docker images`                | 列出镜像      |
| `docker exec -it <id> bash`    | 进入容器      |
| `docker logs -f <id>`          | 查看日志      |
| `docker stop/rm <id>`          | 停止/删除容器 |
| `docker rmi <image>`           | 删除镜像      |

## docker run 常用参数

```bash
docker run -d \              # 后台运行
  --name myapp \             # 容器名
  -p 8080:80 \               # 端口映射
  -v /host:/container \      # 挂载卷
  -e KEY=value \             # 环境变量
  --network mynet \          # 网络
  --restart always \         # 重启策略
  nginx
```

## Dockerfile 指令

```dockerfile
FROM node:18-alpine         # 基础镜像
WORKDIR /app               # 工作目录
COPY . .                   # 复制文件
RUN npm install            # 执行命令
ENV PORT=3000              # 环境变量
EXPOSE 3000                # 暴露端口
CMD ["npm", "start"]       # 启动命令
```

## Docker Compose

```yaml
version: "3.8"
services:
  app:
    build: .
    ports: ["3000:3000"]
    volumes: [".:/app"]
    depends_on: [db]
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret
    volumes: [db-data:/var/lib/postgresql/data]
volumes:
  db-data:
```

## Compose 命令

```bash
docker compose up -d       # 启动
docker compose down -v     # 停止并删除卷
docker compose logs -f     # 查看日志
docker compose ps          # 查看状态
```

## 系统清理

```bash
docker system prune -a     # 清理所有
docker volume prune        # 清理卷
docker network prune       # 清理网络
```
