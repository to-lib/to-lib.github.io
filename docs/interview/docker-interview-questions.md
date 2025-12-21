---
sidebar_position: 11
title: Docker 面试题
description: Docker 常见面试问题与答案
---

# Docker 面试题

## 基础概念

### Q: Docker 镜像和容器有什么区别？

**镜像（Image）**：

- 只读的模板，包含运行应用所需的代码、运行时、库和配置
- 由多个只读层（Layer）组成
- 可以理解为"类"的概念

**容器（Container）**：

- 镜像的运行实例，可读写
- 在镜像的基础上添加一个可写层
- 可以理解为"对象"的概念

```bash
# 镜像是静态的
docker images

# 容器是动态的
docker ps -a
```

---

### Q: Docker 的分层存储是什么？有什么好处？

Docker 镜像采用联合文件系统（UnionFS），由多个只读层叠加而成：

```
┌──────────────────────────────────┐
│   读写层 (Container Layer)        │ ← 容器运行时
├──────────────────────────────────┤
│   应用层 (COPY/ADD)              │
├──────────────────────────────────┤
│   依赖层 (RUN npm install)       │
├──────────────────────────────────┤
│   基础镜像层 (FROM node:18)       │
└──────────────────────────────────┘
```

**好处**：

1. **空间复用**：相同的基础层可被多个镜像共享
2. **构建加速**：未变更的层可使用缓存
3. **快速部署**：仅传输变化的层

---

### Q: Docker 和虚拟机的区别？

| 特性       | Docker 容器                | 虚拟机            |
| ---------- | -------------------------- | ----------------- |
| 虚拟化级别 | 操作系统级（共享内核）     | 硬件级（完整 OS） |
| 启动时间   | 秒级                       | 分钟级            |
| 资源占用   | MB 级                      | GB 级             |
| 性能损耗   | 几乎无                     | 约 5-15%          |
| 隔离性     | 进程级                     | 系统级（更强）    |
| 系统支持   | 仅 Linux（Windows 需 WSL） | 任意 OS           |

---

### Q: CMD 和 ENTRYPOINT 的区别？

```dockerfile
# CMD - 可被 docker run 参数覆盖
CMD ["nginx", "-g", "daemon off;"]

# ENTRYPOINT - 不易被覆盖，适合作为固定入口
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--help"]  # 作为 ENTRYPOINT 的默认参数
```

| 场景                     | CMD      | ENTRYPOINT                |
| ------------------------ | -------- | ------------------------- |
| `docker run image`       | 执行 CMD | 执行 ENTRYPOINT + CMD     |
| `docker run image <cmd>` | 覆盖 CMD | 执行 ENTRYPOINT + `<cmd>` |

**最佳实践**：

```dockerfile
ENTRYPOINT ["java", "-jar", "app.jar"]
CMD ["--spring.profiles.active=prod"]
```

---

### Q: COPY 和 ADD 的区别？

| 特性         | COPY        | ADD                |
| ------------ | ----------- | ------------------ |
| 复制本地文件 | ✅          | ✅                 |
| 自动解压 tar | ❌          | ✅                 |
| 支持 URL     | ❌          | ✅                 |
| 推荐程度     | ✅ 优先使用 | 仅在需要解压时使用 |

```dockerfile
# 推荐
COPY package.json ./

# 仅在需要解压时使用
ADD archive.tar.gz /app/
```

---

## 网络与存储

### Q: Docker 有哪些网络模式？

| 模式      | 说明               | 使用场景       |
| --------- | ------------------ | -------------- |
| `bridge`  | 默认网络，NAT 转发 | 大多数单机场景 |
| `host`    | 共享主机网络       | 高性能需求     |
| `none`    | 无网络             | 安全隔离       |
| `overlay` | 跨主机网络         | Swarm 集群     |
| `macvlan` | 独立 MAC 地址      | 直连物理网络   |

```bash
docker run --network bridge nginx
docker run --network host nginx
docker run --network none alpine
```

---

### Q: 数据卷（Volumes）和绑定挂载（Bind Mounts）的区别？

| 特性     | Volumes                    | Bind Mounts        |
| -------- | -------------------------- | ------------------ |
| 管理方式 | Docker 管理                | 用户管理           |
| 存储位置 | `/var/lib/docker/volumes/` | 任意主机路径       |
| 备份迁移 | 更容易                     | 需手动处理         |
| 适用场景 | 生产环境数据持久化         | 开发环境、配置文件 |

```bash
# Volumes - 推荐用于数据持久化
docker run -v mydata:/data nginx

# Bind Mounts - 适用于开发环境
docker run -v $(pwd)/src:/app/src node
```

---

### Q: 如何实现容器间通信？

**同一网络**（推荐）：

```bash
docker network create mynet
docker run -d --name redis --network mynet redis
docker run -d --name app --network mynet myapp
# app 中可通过 redis:6379 访问
```

**Link（已废弃）**：

```bash
docker run -d --name db mysql
docker run -d --link db:database myapp
```

---

## Dockerfile 最佳实践

### Q: 如何减小镜像体积？

1. **使用 Alpine 基础镜像**

   ```dockerfile
   FROM node:18-alpine  # ~130MB vs node:18 ~1GB
   ```

2. **多阶段构建**

   ```dockerfile
   FROM golang:1.21 AS builder
   RUN go build -o app

   FROM alpine:latest
   COPY --from=builder /app .
   ```

3. **合并 RUN 指令并清理缓存**

   ```dockerfile
   RUN apt-get update && apt-get install -y curl \
       && rm -rf /var/lib/apt/lists/*
   ```

4. **使用 .dockerignore**
   ```plaintext
   node_modules
   .git
   *.md
   ```

---

### Q: 如何优化构建缓存？

**原则**：将变化少的指令放前面

```dockerfile
# ✅ 好的做法
COPY package.json package-lock.json ./
RUN npm ci
COPY . .  # 源码变化不会导致 npm ci 重新执行

# ❌ 不好的做法
COPY . .
RUN npm ci  # 任何文件变化都会重新安装依赖
```

---

## 生产环境

### Q: 如何处理容器日志？

```bash
# 查看日志
docker logs -f --tail 100 container_name

# 配置日志驱动
docker run --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 nginx
```

全局配置 `/etc/docker/daemon.json`：

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

---

### Q: Docker 容器如何实现资源限制？

```bash
# 内存限制
docker run --memory=512m --memory-swap=1g nginx

# CPU 限制
docker run --cpus=1.5 nginx  # 最多使用 1.5 核
docker run --cpu-shares=512 nginx  # 相对权重

# 组合使用
docker run -d \
  --memory=512m \
  --cpus=1.0 \
  --pids-limit=100 \
  nginx
```

---

### Q: 容器健康检查怎么配置？

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
  CMD curl -f http://localhost:8080/health || exit 1
```

```yaml
# docker-compose.yml
services:
  app:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

### Q: 如何排查容器问题？

```bash
# 1. 查看日志
docker logs container_name

# 2. 进入容器
docker exec -it container_name sh

# 3. 查看资源使用
docker stats container_name

# 4. 查看详细信息
docker inspect container_name

# 5. 查看进程
docker top container_name

# 6. 查看网络
docker network inspect bridge
```

---

## Docker Compose

### Q: docker compose up 和 docker compose run 的区别？

| 命令                 | 说明           |
| -------------------- | -------------- |
| `docker compose up`  | 启动所有服务   |
| `docker compose run` | 运行一次性命令 |

```bash
# 启动所有服务
docker compose up -d

# 运行一次性命令（如数据库迁移）
docker compose run --rm app npm run migrate
```

---

### Q: depends_on 能保证服务就绪吗？

**不能**。`depends_on` 只保证启动顺序，不保证服务就绪。

正确做法是使用健康检查：

```yaml
services:
  app:
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
      interval: 10s
      timeout: 5s
      retries: 5
```

---

## 安全相关

### Q: 容器以非 root 用户运行有什么好处？

1. **最小权限原则**：减少攻击面
2. **防止容器逃逸**：限制容器内进程权限
3. **符合合规要求**：许多安全标准要求

```dockerfile
RUN groupadd -r app && useradd -r -g app app
USER app
```

---

### Q: 如何保护敏感信息？

1. **Docker Secrets**（Swarm 模式）

   ```bash
   echo "my_password" | docker secret create db_pass -
   docker service create --secret db_pass myapp
   ```

2. **环境变量文件**

   ```yaml
   services:
     app:
       env_file: .env # 不要提交到版本控制
   ```

3. **构建时使用 BuildKit secrets**
   ```dockerfile
   RUN --mount=type=secret,id=npmrc,target=/root/.npmrc npm ci
   ```
