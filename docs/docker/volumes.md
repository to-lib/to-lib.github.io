---
sidebar_position: 7
title: 数据持久化
description: Docker 数据卷与持久化存储详解
---

# Docker 数据持久化

Docker 容器是临时的，容器删除后数据会丢失。通过数据卷可以实现数据持久化。

## 存储类型

| 类型            | 说明                 | 使用场景           |
| --------------- | -------------------- | ------------------ |
| **Volumes**     | 由 Docker 管理的存储 | 生产环境数据持久化 |
| **Bind Mounts** | 挂载主机目录         | 开发环境、配置文件 |
| **tmpfs**       | 内存存储             | 敏感数据、临时缓存 |

## Volumes（数据卷）

### 创建和管理

```bash
# 创建卷
docker volume create myvolume

# 列出卷
docker volume ls

# 查看卷详情
docker volume inspect myvolume

# 删除卷
docker volume rm myvolume

# 清理未使用卷
docker volume prune
```

### 使用数据卷

```bash
# 使用命名卷
docker run -d -v myvolume:/data nginx

# 自动创建卷
docker run -d -v newvolume:/app/data nginx

# 只读挂载
docker run -d -v myvolume:/data:ro nginx
```

### 共享数据卷

```bash
# 多个容器共享同一卷
docker run -d --name app1 -v shared-data:/data nginx
docker run -d --name app2 -v shared-data:/data nginx
```

### 备份和恢复

```bash
# 备份卷数据
docker run --rm \
  -v myvolume:/data \
  -v $(pwd):/backup \
  alpine tar cvf /backup/backup.tar /data

# 恢复卷数据
docker run --rm \
  -v myvolume:/data \
  -v $(pwd):/backup \
  alpine tar xvf /backup/backup.tar -C /
```

## Bind Mounts（绑定挂载）

### 基本使用

```bash
# 挂载目录
docker run -d -v /host/path:/container/path nginx

# 挂载文件
docker run -d -v /host/config.conf:/etc/app/config.conf nginx

# 只读挂载
docker run -d -v /host/path:/container/path:ro nginx
```

### 开发环境示例

```bash
# 挂载源代码目录（支持热重载）
docker run -d \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/package.json:/app/package.json \
  -p 3000:3000 \
  node:18 npm run dev
```

### 使用 --mount 语法

```bash
# 更明确的语法
docker run -d \
  --mount type=bind,source=/host/path,target=/container/path \
  nginx

# 只读
docker run -d \
  --mount type=bind,source=/host/path,target=/container/path,readonly \
  nginx
```

## tmpfs 挂载

```bash
# 使用 tmpfs
docker run -d \
  --tmpfs /tmp:size=100m \
  nginx

# 使用 --mount
docker run -d \
  --mount type=tmpfs,destination=/tmp,tmpfs-size=100m \
  nginx
```

## Docker Compose 配置

### 命名卷

```yaml
version: "3.8"

services:
  db:
    image: postgres:15
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

### 绑定挂载

```yaml
services:
  app:
    image: node:18
    volumes:
      - ./src:/app/src
      - ./package.json:/app/package.json
```

### 高级配置

```yaml
volumes:
  # 使用驱动选项
  data:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.100,rw
      device: ":/path/to/dir"

  # 使用外部卷
  external-data:
    external: true
    name: my-external-volume
```

### 多种挂载类型

```yaml
services:
  app:
    volumes:
      # 命名卷
      - data:/app/data
      # 绑定挂载
      - ./config:/app/config:ro
      # 匿名卷
      - /app/cache
      # tmpfs
      - type: tmpfs
        target: /app/tmp
        tmpfs:
          size: 100m

volumes:
  data:
```

## 存储驱动

### 配置存储驱动

`/etc/docker/daemon.json`:

```json
{
  "storage-driver": "overlay2"
}
```

### 常用存储驱动

| 驱动         | 说明                            |
| ------------ | ------------------------------- |
| overlay2     | 推荐，适用于大多数 Linux 发行版 |
| aufs         | 旧版 Ubuntu                     |
| devicemapper | RHEL/CentOS 7                   |
| btrfs        | Btrfs 文件系统                  |
| zfs          | ZFS 文件系统                    |

## 最佳实践

### 1. 数据库数据

```yaml
services:
  mysql:
    image: mysql:8.0
    volumes:
      - mysql-data:/var/lib/mysql
    environment:
      MYSQL_ROOT_PASSWORD: secret

volumes:
  mysql-data:
```

### 2. 配置文件

```yaml
services:
  nginx:
    image: nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
```

### 3. 日志持久化

```yaml
services:
  app:
    image: myapp
    volumes:
      - app-logs:/var/log/app

volumes:
  app-logs:
```

### 4. 开发环境

```yaml
services:
  app:
    build: .
    volumes:
      - .:/app
      - node_modules:/app/node_modules # 排除 node_modules
    command: npm run dev

volumes:
  node_modules:
```

## 数据管理命令

```bash
# 查看容器挂载的卷
docker inspect -f '{{json .Mounts}}' container_name

# 复制文件到卷
docker run --rm -v myvolume:/data -v $(pwd):/src alpine cp /src/file.txt /data/

# 查看卷使用情况
docker system df -v
```

## 常见问题

### 权限问题

```dockerfile
# Dockerfile 中设置权限
RUN mkdir -p /data && chown -R 1000:1000 /data
USER 1000
```

```bash
# 运行时指定用户
docker run -u 1000:1000 -v myvolume:/data myimage
```

### 数据同步性能（macOS/Windows）

```yaml
# 使用 cached 或 delegated 模式
services:
  app:
    volumes:
      - ./src:/app/src:cached # 优化读取性能
      - ./data:/app/data:delegated # 优化写入性能
```

### 卷数据迁移

```bash
# 创建新卷
docker volume create newvolume

# 复制数据
docker run --rm \
  -v oldvolume:/from \
  -v newvolume:/to \
  alpine cp -av /from/. /to/
```
