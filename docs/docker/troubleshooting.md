---
sidebar_position: 15
title: 排障指南
description: Docker 常见故障排查思路与命令清单
---

# Docker 排障指南

## 排查总思路

当容器“跑不起来 / 跑不对 / 跑不稳”时，先把问题归类：

- **启动失败**：容器进程退出、健康检查失败、依赖未就绪
- **访问失败**：端口映射/网络/DNS/防火墙/反向代理
- **数据异常**：卷挂载、权限、路径覆盖、初始化脚本
- **性能问题**：CPU/内存/IO/网络瓶颈、资源限制、宿主机压力
- **构建失败**：依赖下载、缓存、平台架构、BuildKit 行为差异

建议始终按以下顺序拿信息：

```bash
# 1) 容器是否在运行？退出码是什么？
docker ps -a

# 2) 容器日志（最重要）
docker logs --tail 200 <container>

# 3) 容器启动命令/配置（端口、卷、网络、环境变量、入口命令）
docker inspect <container>

# 4) 进入容器确认进程、端口、配置文件是否如预期
docker exec -it <container> sh
```

## 启动后立即退出

### 现象

- `docker ps` 看不到容器
- `docker ps -a` 显示 `Exited (x)`

### 排查

```bash
# 查看退出码与最后状态
docker ps -a --no-trunc

docker logs <container>

docker inspect --format='{{.State.Status}} {{.State.ExitCode}} {{.State.Error}}' <container>
```

高频原因：

- **主进程执行完就退出**（容器必须有前台进程）
- **入口命令/配置错误**
- **依赖服务没起来**（数据库、MQ 等）
- **权限/文件缺失**导致进程启动即崩溃

## 端口映射与访问失败

### 先确认端口是否正确暴露

```bash
# 宿主机->容器端口映射
docker port <container>

# 只绑定到 127.0.0.1 时，外部机器无法访问
# 示例：127.0.0.1:8080->80/tcp
```

### 在容器内确认服务是否监听

```bash
# Alpine 常用
ss -lntp || netstat -lntp

# 直接从容器内访问服务
curl -v http://127.0.0.1:<port>
```

### 常见坑

- 容器内服务只监听 `127.0.0.1`，导致端口映射后仍不可达（应监听 `0.0.0.0`）
- 端口映射写反：`-p 主机端口:容器端口`
- 宿主机防火墙/安全组拦截
- 反向代理（Nginx）上游地址写错（容器名、端口、网络不一致）

## DNS / 网络问题

### 容器能否解析域名

```bash
# 快速验证 DNS
docker run --rm busybox nslookup example.com
```

### 容器间是否可互通

```bash
docker network ls

docker network inspect <network>

# 在同一自定义网络内可通过容器名解析
docker exec -it <container> ping <other-container>
docker exec -it <container> curl -v http://<other-container>:<port>
```

### 典型问题

- **使用默认 `bridge` 网络**时，容器名解析不稳定/不支持（推荐自定义 bridge 网络）
- Compose 中不同 `project` 或不同 `network` 导致容器不可见
- DNS 被公司网络劫持/限制：可在 `daemon.json` 配置 `dns` 或在 `docker run --dns ...` 指定

## 卷挂载与数据异常

### 挂载覆盖导致“文件不见了”

当你把宿主机目录挂到容器路径（例如 `/app`）时，会覆盖镜像里同路径的内容。

```bash
# 查看挂载点
docker inspect -f '{{json .Mounts}}' <container>
```

### 权限问题

```bash
# 进入容器查看文件属主
docker exec -it <container> sh
ls -la <path>

# 宿主机目录权限/属主可能需要调整
# 或者运行时指定用户（与宿主机目录 UID/GID 对齐）
docker run -u 1000:1000 -v $(pwd)/data:/data <image>
```

### 数据卷备份/恢复

```bash
# 备份卷
docker run --rm -v <volume>:/data -v $(pwd):/backup alpine tar -cvf /backup/backup.tar /data

# 恢复卷
docker run --rm -v <volume>:/data -v $(pwd):/backup alpine tar -xvf /backup/backup.tar -C /
```

## 资源与性能

### 快速观察

```bash
# 资源概览
docker stats

# 容器进程
docker top <container>

# 宿主机层面（Linux）
# top / htop / iostat / vmstat / sar
```

### OOM / 内存被杀

```bash
# ExitCode=137 常见于 OOM Kill
docker inspect --format='{{.State.OOMKilled}} {{.State.ExitCode}}' <container>
```

处理方向：

- 增加内存限制或取消过小限制
- 排查应用内存泄漏/缓存策略
- 合理设置 `--memory`/`--memory-swap`，避免无界使用

## 构建失败排查

```bash
# 获取详细构建日志
DOCKER_BUILDKIT=1 docker build --progress=plain .

# 关闭缓存验证问题是否来自缓存
DOCKER_BUILDKIT=1 docker build --no-cache .
```

常见原因：

- 依赖下载失败（网络、代理、证书）
- 多平台构建（`linux/arm64` vs `linux/amd64`）导致二进制不可用
- `COPY` 路径与 `.dockerignore` 排除规则冲突

## Docker Compose 常见排障

```bash
# 服务状态
docker compose ps

# 查看日志（按服务）
docker compose logs --tail 200 -f <service>

# 查看最终生效的配置（合并 override / env 替换后）
docker compose config

# 在某个服务容器里执行命令
docker compose exec <service> sh
```

重点检查：

- `depends_on` 只保证启动顺序，不保证就绪（建议配合 `healthcheck`）
- `.env` 文件是否在 compose 文件同目录
- 网络是否同一 `project`（`COMPOSE_PROJECT_NAME` 会影响网络/容器前缀）

## 常用“一键信息收集”命令

```bash
# 1) 最近退出的容器
CID=$(docker ps -aq --latest)

# 2) 打印关键信息
docker ps -a --no-trunc | head

docker logs --tail 200 "$CID"

docker inspect "$CID" --format='{{json .State}}'

docker inspect "$CID" --format='{{json .HostConfig}}'
```
