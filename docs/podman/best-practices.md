---
sidebar_position: 10
title: 最佳实践
description: Podman 生产环境最佳实践
---

# Podman 最佳实践

本文总结 Podman 在生产环境中的最佳实践，帮助你构建安全、高效、易维护的容器化应用。

## 容器运行

### 健康检查

```bash
# 运行时指定健康检查
podman run -d \
  --health-cmd="curl -f http://localhost/ || exit 1" \
  --health-interval=30s \
  --health-retries=3 \
  --health-start-period=10s \
  --health-timeout=5s \
  nginx

# 查看健康状态
podman inspect --format='{{.State.Health.Status}}' container_name

# Dockerfile 中定义
# HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
#   CMD curl -f http://localhost/ || exit 1
```

### 容器重启策略

```bash
# 使用 systemd 管理（推荐）
podman generate systemd --new --name web > ~/.config/systemd/user/web.service
systemctl --user enable --now web

# 或使用 Quadlet（Podman 4.4+）
cat > ~/.config/containers/systemd/web.container << EOF
[Container]
Image=nginx
PublishPort=8080:80

[Service]
Restart=always

[Install]
WantedBy=default.target
EOF
```

### 资源限制

```bash
# 推荐的生产配置
podman run -d \
  --name app \
  --memory=1g \
  --memory-reservation=512m \
  --cpus=1.0 \
  --pids-limit=100 \
  --ulimit nofile=65535:65535 \
  myapp
```

## 日志管理

### 日志驱动配置

```bash
# 使用 journald 日志驱动
podman run -d --log-driver=journald nginx

# 使用 json-file（默认）
podman run -d --log-opt max-size=10m --log-opt max-file=3 nginx

# 禁用日志（临时容器）
podman run --log-driver=none alpine echo "test"
```

### 日志查看

```bash
# 查看日志
podman logs container_name

# 实时跟踪
podman logs -f container_name

# 使用 journald
journalctl --user CONTAINER_NAME=container_name

# 按时间过滤
podman logs --since "2024-01-01" container_name
```

## 镜像管理

### 镜像标签策略

```bash
# 使用语义化版本
myapp:1.0.0
myapp:1.0
myapp:1
myapp:latest

# 使用 Git SHA
myapp:sha-abc1234

# 使用日期标签
myapp:2024-01-15
```

### 自动清理

```bash
# 清理未使用的镜像
podman image prune -a

# 清理所有未使用资源
podman system prune -a --volumes

# 定期清理（添加到 cron）
0 2 * * 0 podman system prune -a -f --volumes
```

## 网络最佳实践

### 使用自定义网络

```bash
# 创建专用网络
podman network create --subnet 10.89.0.0/24 app-network

# 应用使用专用网络
podman run -d --network app-network --name db postgres
podman run -d --network app-network --name app myapp
```

### 仅暴露必要端口

```bash
# 仅本地访问
podman run -p 127.0.0.1:8080:80 nginx

# 不要使用 -P（暴露所有端口）
```

## 存储最佳实践

### 使用命名卷

```bash
# 创建命名卷
podman volume create app-data

# 使用命名卷
podman run -v app-data:/data myapp

# 查看卷位置
podman volume inspect app-data
```

### 备份策略

```bash
# 备份卷数据
podman run --rm -v app-data:/data:ro -v $(pwd):/backup alpine \
  tar czf /backup/app-data-backup.tar.gz -C /data .

# 恢复卷数据
podman run --rm -v app-data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/app-data-backup.tar.gz -C /data
```

## 安全最佳实践

### 运行时安全配置

```bash
podman run -d \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid \
  --cap-drop=all \
  --cap-add=NET_BIND_SERVICE \
  --security-opt=no-new-privileges \
  --user 1000:1000 \
  nginx
```

### 使用非 root 用户

```dockerfile
FROM node:18-alpine

# 创建非 root 用户
RUN addgroup -g 1001 appgroup && \
    adduser -u 1001 -G appgroup -S appuser

WORKDIR /app
COPY --chown=appuser:appgroup . .

USER appuser

CMD ["node", "server.js"]
```

## Rootless 生产配置

### 持久化服务

```bash
# 启用 linger（用户服务持续运行）
loginctl enable-linger $USER

# 配置 systemd 用户服务
mkdir -p ~/.config/systemd/user/
podman generate systemd --new --name app > ~/.config/systemd/user/app.service
systemctl --user daemon-reload
systemctl --user enable --now app
```

### 资源配置

```bash
# 配置足够的 subuid/subgid
sudo usermod --add-subuids 100000-165535 $USER
sudo usermod --add-subgids 100000-165535 $USER
```

## 监控与告警

### 资源监控

```bash
# 实时资源使用
podman stats

# 查看特定容器
podman stats --no-stream container_name

# 格式化输出
podman stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### 集成 Prometheus

```bash
# 启用 Podman API
systemctl --user enable --now podman.socket

# 使用 cAdvisor 或 Prometheus exporter 收集指标
podman run -d \
  --volume=/:/rootfs:ro \
  --volume=/var/run/podman/podman.sock:/var/run/docker.sock:ro \
  --publish=8080:8080 \
  gcr.io/cadvisor/cadvisor
```

## CI/CD 集成

### Jenkins 集成

```groovy
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'podman build -t myapp:${BUILD_NUMBER} .'
            }
        }
        stage('Push') {
            steps {
                sh 'podman push myapp:${BUILD_NUMBER} registry.example.com/myapp:${BUILD_NUMBER}'
            }
        }
    }
}
```

### GitHub Actions

```yaml
- name: Build with Podman
  run: |
    podman build -t myapp:${{ github.sha }} .
    podman push myapp:${{ github.sha }} ghcr.io/${{ github.repository }}:${{ github.sha }}
```

## 故障排查

### 常用调试命令

```bash
# 查看容器详情
podman inspect container_name

# 进入容器调试
podman exec -it container_name /bin/sh

# 导出容器文件系统
podman export container_name -o container.tar

# 查看容器进程
podman top container_name

# 查看端口映射
podman port container_name
```

## 清单总结

| 类别 | 最佳实践                             |
| ---- | ------------------------------------ |
| 运行 | 使用 Rootless、健康检查、资源限制    |
| 安全 | 最小权限、只读文件系统、非 root 用户 |
| 存储 | 命名卷、定期备份                     |
| 网络 | 自定义网络、仅暴露必要端口           |
| 日志 | journald 集成、日志轮转              |
| 监控 | 资源监控、Prometheus 集成            |
| 服务 | systemd/Quadlet、enable-linger       |
