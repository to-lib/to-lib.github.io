---
sidebar_position: 35
title: Docker Bench
description: CIS Docker 安全基准检测脚本
---

# Docker Bench

Docker Bench for Security 是一个脚本，用于检查 Docker 部署是否符合 CIS（Center for Internet Security）安全基准。

## 快速使用

```bash
# 运行 Docker Bench
docker run --rm --net host --pid host --userns host --cap-add audit_control \
  -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
  -v /etc:/etc:ro \
  -v /lib/systemd/system:/lib/systemd/system:ro \
  -v /usr/bin/containerd:/usr/bin/containerd:ro \
  -v /usr/bin/runc:/usr/bin/runc:ro \
  -v /usr/lib/systemd:/usr/lib/systemd:ro \
  -v /var/lib:/var/lib:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  --label docker_bench_security \
  docker/docker-bench-security
```

## 检查类别

### 1. 主机配置

| 检查项 | 说明 |
|--------|------|
| 1.1 | 为容器创建单独分区 |
| 1.2 | 加固容器主机 |
| 1.3 | 保持 Docker 更新 |
| 1.4 | 只允许受信任用户控制 Docker |
| 1.5 | 审计 Docker 守护进程 |
| 1.6 | 审计 Docker 文件和目录 |

### 2. Docker 守护进程配置

| 检查项 | 说明 |
|--------|------|
| 2.1 | 限制默认网桥上的容器间通信 |
| 2.2 | 设置日志级别为 info |
| 2.3 | 允许 Docker 修改 iptables |
| 2.4 | 不使用不安全的 registry |
| 2.5 | 不使用 aufs 存储驱动 |
| 2.6 | 配置 TLS 认证 |
| 2.7 | 设置默认 ulimit |
| 2.8 | 启用用户命名空间 |
| 2.9 | 使用默认 cgroup |
| 2.10 | 设置默认 seccomp 配置 |
| 2.11 | 启用实验性功能时要谨慎 |
| 2.12 | 限制容器获取新权限 |
| 2.13 | 启用实时恢复 |
| 2.14 | 禁用 Swarm 模式（如不使用） |

### 3. Docker 守护进程配置文件

| 检查项 | 说明 |
|--------|------|
| 3.1-3.22 | 验证配置文件权限（644 或更严格） |

### 4. 容器镜像和构建

| 检查项 | 说明 |
|--------|------|
| 4.1 | 创建容器用户 |
| 4.2 | 使用受信任的基础镜像 |
| 4.3 | 不安装不必要的包 |
| 4.4 | 扫描镜像漏洞 |
| 4.5 | 启用 Content Trust |
| 4.6 | 添加 HEALTHCHECK |
| 4.7 | 不使用 update 指令 |
| 4.8 | 删除 setuid/setgid 权限 |
| 4.9 | 使用 COPY 而非 ADD |
| 4.10 | 不存储密钥 |
| 4.11 | 只安装已验证的包 |

### 5. 容器运行时

| 检查项 | 说明 |
|--------|------|
| 5.1 | 不禁用 AppArmor |
| 5.2 | 设置 SELinux 选项 |
| 5.3 | 限制 Linux Capabilities |
| 5.4 | 不使用特权容器 |
| 5.5 | 不挂载敏感主机目录 |
| 5.6 | 不在容器内运行 sshd |
| 5.7 | 不映射特权端口 |
| 5.8 | 只打开需要的端口 |
| 5.9 | 不共享主机网络命名空间 |
| 5.10 | 限制内存使用 |
| 5.11 | 设置 CPU 优先级 |
| 5.12 | 挂载根文件系统为只读 |
| 5.13 | 限制容器重启次数 |
| 5.14 | 不共享主机进程命名空间 |
| 5.15 | 不共享主机 IPC 命名空间 |
| 5.16 | 不直接暴露主机设备 |
| 5.17 | 覆盖默认 ulimit |
| 5.18 | 不共享主机 UTS 命名空间 |
| 5.19 | 不禁用默认 seccomp |
| 5.20 | 不使用 docker exec --privileged |
| 5.21 | 不使用 docker exec --user=root |
| 5.22 | 不禁用默认 cgroup |
| 5.23 | 限制容器获取额外权限 |
| 5.24 | 检查容器健康状态 |
| 5.25 | 限制 PIDs |
| 5.26 | 不使用 host 网络模式 |
| 5.27 | 限制 CPU 使用 |
| 5.28 | 使用 PIDs cgroup 限制 |

### 6. Docker 安全操作

| 检查项 | 说明 |
|--------|------|
| 6.1 | 定期审计镜像 |
| 6.2 | 定期审计容器 |

## 输出示例

```
[INFO] 1 - Host Configuration
[PASS] 1.1  - Ensure a separate partition for containers has been created
[WARN] 1.2  - Ensure the container host has been Hardened
[PASS] 1.3  - Ensure Docker is up to date

[INFO] 2 - Docker daemon configuration
[WARN] 2.1  - Ensure network traffic is restricted between containers
[PASS] 2.2  - Ensure the logging level is set to 'info'
[PASS] 2.3  - Ensure Docker is allowed to make changes to iptables
```

## 修复常见问题

### 2.1 限制容器间通信

```json
// /etc/docker/daemon.json
{
  "icc": false
}
```

### 2.14 禁用 Swarm（如不使用）

```bash
docker swarm leave --force
```

### 4.1 创建非 root 用户

```dockerfile
RUN useradd -r -u 1001 appuser
USER appuser
```

### 5.3 限制 Capabilities

```bash
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE nginx
```

### 5.10 限制内存

```bash
docker run --memory=512m --memory-swap=512m nginx
```

### 5.12 只读根文件系统

```bash
docker run --read-only --tmpfs /tmp nginx
```

## 自动化检查

### CI/CD 集成

```yaml
# GitHub Actions
- name: Run Docker Bench
  run: |
    docker run --rm \
      -v /var/run/docker.sock:/var/run/docker.sock \
      docker/docker-bench-security \
      -l /dev/stdout 2>&1 | tee bench-results.txt
    
    # 检查是否有 WARN
    if grep -q "\[WARN\]" bench-results.txt; then
      echo "Security warnings found!"
      exit 1
    fi
```

### 定期扫描脚本

```bash
#!/bin/bash
# security-scan.sh

REPORT_DIR="/var/log/docker-bench"
mkdir -p $REPORT_DIR

docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $REPORT_DIR:/report \
  docker/docker-bench-security \
  -l /report/$(date +%Y%m%d).log

# 发送告警（如有 WARN）
if grep -q "\[WARN\]" $REPORT_DIR/$(date +%Y%m%d).log; then
  # 发送邮件或 Slack 通知
  echo "Docker security warnings detected"
fi
```

## 相关工具

| 工具 | 用途 |
|------|------|
| Docker Bench | CIS 基准检测 |
| Trivy | 镜像漏洞扫描 |
| Docker Scout | 官方安全扫描 |
| Falco | 运行时安全监控 |
| Anchore | 镜像策略检查 |
