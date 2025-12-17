---
sidebar_position: 9
title: 安全配置
description: Podman 安全特性与配置详解
---

# Podman 安全配置

Podman 的设计理念就是安全优先，提供多层安全机制保护容器环境。

## Rootless 安全优势

Rootless 模式是 Podman 最重要的安全特性：

```mermaid
graph LR
    A[用户进程] --> B[Podman]
    B --> C[用户命名空间]
    C --> D[容器]
    D --> E[受限的系统调用]
```

### 优势

| 特性             | Rootless | Root |
| ---------------- | -------- | ---- |
| 宿主机 root 权限 | 无       | 有   |
| 容器逃逸风险     | 极低     | 较高 |
| 系统文件访问     | 受限     | 完全 |
| 网络权限         | 受限     | 完全 |

```bash
# 验证 rootless 模式
podman info | grep rootless
```

## 用户命名空间

用户命名空间提供 UID/GID 隔离：

```bash
# 查看用户映射
podman unshare cat /proc/self/uid_map

# 配置 subuid/subgid
sudo usermod --add-subuids 100000-165535 $USER
sudo usermod --add-subgids 100000-165535 $USER
```

### 用户映射工作原理

容器内的 root (UID 0) 映射到主机上的普通用户：

```bash
# 容器内是 root
podman run --rm alpine id
# uid=0(root) gid=0(root)

# 但在主机上是普通用户
podman top $(podman run -d alpine sleep 100) huser
```

## SELinux 集成

Podman 与 SELinux 深度集成：

```bash
# 查看 SELinux 状态
getenforce

# 容器默认运行在受限上下文
podman run --rm alpine cat /proc/self/attr/current

# 禁用 SELinux 标签（不推荐）
podman run --security-opt label=disable alpine
```

### 卷挂载与 SELinux

```bash
# 私有标签（推荐，单容器访问）
podman run -v /data:/data:Z nginx

# 共享标签（多容器访问）
podman run -v /data:/data:z nginx
```

## AppArmor 支持

在使用 AppArmor 的系统上：

```bash
# 查看默认配置文件
cat /etc/apparmor.d/containers/podman

# 使用自定义配置
podman run --security-opt apparmor=my-profile alpine
```

## Seccomp 配置

Seccomp 限制容器可用的系统调用：

```bash
# 使用默认 seccomp 配置（推荐）
podman run alpine

# 查看默认配置
cat /usr/share/containers/seccomp.json

# 使用自定义配置
podman run --security-opt seccomp=custom.json alpine

# 禁用 seccomp（不推荐）
podman run --security-opt seccomp=unconfined alpine
```

### 自定义 Seccomp 配置

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": ["read", "write", "exit", "exit_group"],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

## Capabilities 管理

精细控制容器权限：

```bash
# 查看默认 capabilities
podman run --rm alpine cat /proc/self/status | grep Cap

# 删除所有 capabilities
podman run --cap-drop=all alpine

# 只添加需要的
podman run --cap-drop=all --cap-add=NET_BIND_SERVICE nginx

# 添加特定 capability
podman run --cap-add=SYS_PTRACE alpine
```

### 常用 Capabilities

| Capability         | 说明                 |
| ------------------ | -------------------- |
| `NET_BIND_SERVICE` | 绑定低于 1024 的端口 |
| `SYS_PTRACE`       | 进程追踪调试         |
| `SYS_ADMIN`        | 系统管理（危险）     |
| `NET_ADMIN`        | 网络管理             |
| `SETUID/SETGID`    | 改变用户/组 ID       |

## 只读容器

增强容器安全性：

```bash
# 只读根文件系统
podman run --read-only nginx

# 只读但允许临时目录
podman run --read-only --tmpfs /tmp nginx

# 只读挂载卷
podman run -v /config:/config:ro nginx
```

## 资源限制

使用 cgroups 限制容器资源：

```bash
# 限制内存
podman run --memory=512m nginx

# 限制 CPU
podman run --cpus=0.5 nginx          # 50% CPU
podman run --cpu-shares=512 nginx    # 相对权重

# 限制进程数
podman run --pids-limit=100 nginx

# 限制 I/O
podman run --device-read-bps=/dev/sda:1mb nginx
```

## 网络安全

```bash
# 禁用网络
podman run --network none alpine

# 限制端口暴露范围
podman run -p 127.0.0.1:8080:80 nginx  # 仅本地访问
```

## 镜像安全

### 镜像签名验证

```bash
# 配置签名策略
cat /etc/containers/policy.json

# 要求签名
# 在 policy.json 中配置 "signedBy" 要求
```

### 镜像扫描

```bash
# 使用 Trivy 扫描
trivy image nginx:alpine

# 扫描本地镜像
podman images --format "{{.Repository}}:{{.Tag}}" | xargs -I {} trivy image {}
```

## 安全加固清单

### 运行时安全

```bash
# 推荐的安全运行配置
podman run -d \
  --read-only \
  --tmpfs /tmp \
  --cap-drop=all \
  --cap-add=NET_BIND_SERVICE \
  --security-opt=no-new-privileges \
  --user 1000:1000 \
  --memory=512m \
  --pids-limit=50 \
  nginx
```

### 配置检查

```bash
# 查看容器安全配置
podman inspect --format='{{.HostConfig.SecurityOpt}}' container_name

# 查看 capabilities
podman inspect --format='{{.HostConfig.CapAdd}} {{.HostConfig.CapDrop}}' container_name
```

## 审计与日志

```bash
# 查看容器事件
podman events

# 过滤事件
podman events --filter event=start --filter container=web

# 查看容器日志
podman logs --timestamps container_name
```

## 最佳实践

1. **始终使用 Rootless 模式** 除非有特殊需求
2. **遵循最小权限原则**，使用 `--cap-drop=all` 然后只添加需要的
3. **使用只读文件系统** `--read-only`
4. **限制资源使用** 防止资源耗尽攻击
5. **定期扫描镜像** 检查已知漏洞
6. **使用非 root 用户** 在容器内运行应用
7. **启用安全策略** (SELinux/AppArmor)
8. **使用 `--security-opt=no-new-privileges`** 防止提权
