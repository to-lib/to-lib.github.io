---
sidebar_position: 5
title: Docker Compose
description: Docker Compose 多容器编排详解
---

# Docker Compose

Docker Compose 是一个用于定义和运行多容器 Docker 应用程序的工具。

## 安装

### Linux

```bash
# 使用 Docker 插件（推荐）
sudo apt-get install docker-compose-plugin

# 验证安装
docker compose version
```

### macOS/Windows

Docker Desktop 已包含 Docker Compose。

## 基本结构

```yaml
# docker-compose.yml
version: "3.8"

services:
  web:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./html:/usr/share/nginx/html
    networks:
      - webnet

  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: secret
    volumes:
      - db-data:/var/lib/mysql
    networks:
      - webnet

volumes:
  db-data:

networks:
  webnet:
```

## 常用配置

### 服务配置

```yaml
services:
  app:
    # 使用镜像
    image: node:18-alpine

    # 或构建镜像
    build:
      context: .
      dockerfile: Dockerfile
      args:
        NODE_ENV: production

    # 容器名称
    container_name: my-app

    # 重启策略
    restart: always # no, on-failure, unless-stopped

    # 端口映射
    ports:
      - "3000:3000"
      - "127.0.0.1:3001:3001"

    # 环境变量
    environment:
      - NODE_ENV=production
      - DB_HOST=db

    # 从文件加载环境变量
    env_file:
      - .env
      - .env.local

    # 挂载卷
    volumes:
      - ./src:/app/src
      - node_modules:/app/node_modules

    # 依赖关系
    depends_on:
      - db
      - redis

    # 网络
    networks:
      - frontend
      - backend

    # 命令
    command: npm start

    # 入口点
    entrypoint: ["docker-entrypoint.sh"]

    # 工作目录
    working_dir: /app

    # 用户
    user: "1000:1000"
```

### 健康检查

```yaml
services:
  web:
    image: nginx
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 资源限制

```yaml
services:
  app:
    image: node:18
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M
        reservations:
          cpus: "0.25"
          memory: 256M
```

### 网络配置

```yaml
services:
  web:
    networks:
      frontend:
        ipv4_address: 172.16.238.10
      backend:

networks:
  frontend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.16.238.0/24
  backend:
    driver: bridge
```

### 卷配置

```yaml
volumes:
  # 命名卷
  db-data:
    driver: local

  # 使用驱动选项
  logs:
    driver: local
    driver_opts:
      type: none
      device: /var/log/app
      o: bind

services:
  db:
    volumes:
      # 命名卷
      - db-data:/var/lib/mysql
      # 绑定挂载
      - ./config:/etc/mysql/conf.d
      # 只读
      - ./static:/app/static:ro
```

## 实用示例

### Web 应用 + 数据库

```yaml
version: "3.8"

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres-data:
```

### Nginx 反向代理

```yaml
version: "3.8"

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app

  app:
    build: .
    expose:
      - "3000"
```

### 开发环境

```yaml
version: "3.8"

services:
  app:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - node_modules:/app/node_modules
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
    command: npm run dev

  db:
    image: mysql:8.0
    ports:
      - "3306:3306"
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: dev_db
    volumes:
      - mysql-data:/var/lib/mysql

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  node_modules:
  mysql-data:
```

## 常用命令

```bash
# 启动服务
docker compose up

# 后台启动
docker compose up -d

# 指定配置文件
docker compose -f docker-compose.prod.yml up -d

# 构建镜像
docker compose build

# 不使用缓存构建
docker compose build --no-cache

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs

# 实时查看日志
docker compose logs -f

# 查看特定服务日志
docker compose logs app

# 停止服务
docker compose stop

# 停止并删除容器
docker compose down

# 删除卷
docker compose down -v

# 删除镜像
docker compose down --rmi all

# 重启服务
docker compose restart

# 执行命令
docker compose exec app bash

# 运行一次性命令
docker compose run --rm app npm test

# 扩展服务
docker compose up -d --scale app=3
```

## 环境变量

### .env 文件

```plaintext
# .env
COMPOSE_PROJECT_NAME=myproject
DB_PASSWORD=secret123
APP_PORT=3000
```

### 在 compose 文件中使用

```yaml
services:
  app:
    ports:
      - "${APP_PORT:-3000}:3000"
    environment:
      - DB_PASSWORD=${DB_PASSWORD}
```

## 多环境配置

### docker-compose.override.yml

开发环境自动加载：

```yaml
# docker-compose.yml - 基础配置
services:
  app:
    image: myapp:latest

# docker-compose.override.yml - 开发配置（自动加载）
services:
  app:
    build: .
    volumes:
      - .:/app
```

### 生产环境

```yaml
# docker-compose.prod.yml
services:
  app:
    image: myapp:${VERSION:-latest}
    restart: always
    deploy:
      replicas: 3
```

启动：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```
