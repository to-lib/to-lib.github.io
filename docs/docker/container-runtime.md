---
sidebar_position: 17
title: 容器运行时
description: Docker 底层运行时原理 - containerd、runc 与 OCI 标准
---

# 容器运行时

理解 Docker 底层的容器运行时架构，有助于深入掌握容器技术原理。

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                  Docker CLI                      │
└─────────────────────┬───────────────────────────┘
                      │ REST API
┌─────────────────────▼───────────────────────────┐
│               Docker Daemon (dockerd)            │
│  ┌─────────────────────────────────────────┐    │
│  │         Image Management                 │    │
│  │         Network Management               │    │
│  │         Volume Management                │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────┘
                      │ gRPC
┌─────────────────────▼───────────────────────────┐
│                  containerd                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐    │
│  │ Images   │ │Containers│ │   Snapshots  │    │
│  └──────────┘ └──────────┘ └──────────────┘    │
└─────────────────────┬───────────────────────────┘
                      │ OCI Runtime Spec
┌─────────────────────▼───────────────────────────┐
│                    runc                          │
│         (OCI Runtime Implementation)             │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              Linux Kernel                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐    │
│  │ Namespaces│ │ cgroups  │ │   seccomp    │    │
│  └──────────┘ └──────────┘ └──────────────┘    │
└─────────────────────────────────────────────────┘
```

## OCI 标准

OCI (Open Container Initiative) 定义了容器的开放标准。

### OCI 镜像规范

```json
// 镜像 manifest 示例
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "digest": "sha256:abc123...",
    "size": 7023
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "digest": "sha256:def456...",
      "size": 32654
    }
  ]
}
```

### OCI 运行时规范

```json
// config.json - 容器运行时配置
{
  "ociVersion": "1.0.2",
  "process": {
    "terminal": true,
    "user": { "uid": 0, "gid": 0 },
    "args": ["sh"],
    "env": ["PATH=/usr/bin:/bin"],
    "cwd": "/"
  },
  "root": {
    "path": "rootfs",
    "readonly": true
  },
  "hostname": "container",
  "linux": {
    "namespaces": [
      { "type": "pid" },
      { "type": "network" },
      { "type": "ipc" },
      { "type": "uts" },
      { "type": "mount" }
    ]
  }
}
```

## containerd

containerd 是工业级容器运行时，负责镜像传输、容器执行、存储和网络。

### 架构组件

| 组件 | 功能 |
|------|------|
| **Content Store** | 存储镜像层内容 |
| **Snapshotter** | 管理文件系统快照 |
| **Runtime** | 容器生命周期管理 |
| **Metadata Store** | 存储容器元数据 |

### 直接使用 containerd

```bash
# 使用 ctr 命令行工具
# 拉取镜像
ctr images pull docker.io/library/nginx:latest

# 列出镜像
ctr images ls

# 运行容器
ctr run -d docker.io/library/nginx:latest nginx

# 列出容器
ctr containers ls

# 查看任务（运行中的容器）
ctr tasks ls

# 进入容器
ctr tasks exec --exec-id shell nginx sh
```

### nerdctl - containerd 的 Docker 兼容 CLI

```bash
# 安装 nerdctl
brew install nerdctl  # macOS
# 或从 GitHub 下载

# 使用方式与 docker 命令兼容
nerdctl run -d -p 80:80 nginx
nerdctl ps
nerdctl images
nerdctl compose up -d
```

## runc

runc 是 OCI 运行时规范的参考实现，负责实际创建和运行容器。

### 手动使用 runc

```bash
# 创建容器 bundle
mkdir -p mycontainer/rootfs

# 导出镜像文件系统
docker export $(docker create busybox) | tar -C mycontainer/rootfs -xf -

# 生成运行时配置
cd mycontainer
runc spec

# 运行容器
sudo runc run mycontainer

# 列出容器
sudo runc list

# 删除容器
sudo runc delete mycontainer
```

### runc spec 生成的配置

```json
{
  "ociVersion": "1.0.2",
  "process": {
    "terminal": true,
    "user": { "uid": 0, "gid": 0 },
    "args": ["sh"],
    "env": [
      "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
      "TERM=xterm"
    ],
    "cwd": "/",
    "capabilities": {
      "bounding": ["CAP_AUDIT_WRITE", "CAP_KILL", "CAP_NET_BIND_SERVICE"],
      "effective": ["CAP_AUDIT_WRITE", "CAP_KILL", "CAP_NET_BIND_SERVICE"]
    },
    "rlimits": [
      { "type": "RLIMIT_NOFILE", "hard": 1024, "soft": 1024 }
    ]
  },
  "root": {
    "path": "rootfs",
    "readonly": true
  },
  "linux": {
    "resources": {
      "memory": { "limit": 536870912 },
      "cpu": { "shares": 1024 }
    },
    "namespaces": [
      { "type": "pid" },
      { "type": "network" },
      { "type": "ipc" },
      { "type": "uts" },
      { "type": "mount" },
      { "type": "cgroup" }
    ],
    "maskedPaths": [
      "/proc/acpi",
      "/proc/kcore",
      "/sys/firmware"
    ],
    "readonlyPaths": [
      "/proc/bus",
      "/proc/fs",
      "/proc/irq"
    ]
  }
}
```

## cgroups v2

cgroups (Control Groups) 用于限制、记录和隔离进程组的资源使用。

### cgroups v1 vs v2

| 特性 | cgroups v1 | cgroups v2 |
|------|------------|------------|
| 层级结构 | 多层级（每个控制器独立） | 统一层级 |
| 资源控制 | 分散在不同子系统 | 统一接口 |
| 线程支持 | 有限 | 完整支持 |
| 压力监控 | 无 | PSI (Pressure Stall Information) |

### 检查 cgroups 版本

```bash
# 检查系统使用的 cgroups 版本
mount | grep cgroup

# cgroups v2 输出
# cgroup2 on /sys/fs/cgroup type cgroup2

# cgroups v1 输出
# cgroup on /sys/fs/cgroup/memory type cgroup
```

### cgroups v2 资源控制

```bash
# 查看 cgroup 控制器
cat /sys/fs/cgroup/cgroup.controllers
# cpu io memory pids

# 创建 cgroup
sudo mkdir /sys/fs/cgroup/mygroup

# 启用控制器
echo "+cpu +memory +io" | sudo tee /sys/fs/cgroup/mygroup/cgroup.subtree_control

# 设置内存限制
echo "536870912" | sudo tee /sys/fs/cgroup/mygroup/memory.max

# 设置 CPU 限制 (50% of one CPU)
echo "50000 100000" | sudo tee /sys/fs/cgroup/mygroup/cpu.max

# 将进程加入 cgroup
echo $PID | sudo tee /sys/fs/cgroup/mygroup/cgroup.procs
```

### Docker 中的 cgroups

```bash
# 查看容器的 cgroup
docker inspect --format '{{.HostConfig.CgroupParent}}' container_name

# 查看容器资源限制
cat /sys/fs/cgroup/docker/<container_id>/memory.max
cat /sys/fs/cgroup/docker/<container_id>/cpu.max

# 配置 Docker 使用 cgroups v2
# /etc/docker/daemon.json
{
  "default-cgroupns-mode": "private",
  "cgroup-parent": "docker.slice"
}
```

## Linux Namespaces

Namespaces 提供进程级别的隔离。

### Namespace 类型

| Namespace | 隔离内容 | 标志 |
|-----------|----------|------|
| **Mount** | 文件系统挂载点 | CLONE_NEWNS |
| **UTS** | 主机名和域名 | CLONE_NEWUTS |
| **IPC** | 进程间通信 | CLONE_NEWIPC |
| **PID** | 进程 ID | CLONE_NEWPID |
| **Network** | 网络设备、端口 | CLONE_NEWNET |
| **User** | 用户和组 ID | CLONE_NEWUSER |
| **Cgroup** | Cgroup 根目录 | CLONE_NEWCGROUP |

### 查看容器 Namespaces

```bash
# 获取容器 PID
PID=$(docker inspect --format '{{.State.Pid}}' container_name)

# 查看 namespaces
ls -la /proc/$PID/ns/

# 进入容器 namespace
sudo nsenter -t $PID -n ip addr  # 网络 namespace
sudo nsenter -t $PID -m ls /     # 挂载 namespace
sudo nsenter -t $PID -p -r ps    # PID namespace
```

## 其他 OCI 运行时

### gVisor (runsc)

Google 开发的用户态内核，提供更强的隔离。

```bash
# 安装 gVisor
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list
sudo apt-get update && sudo apt-get install -y runsc

# 配置 Docker 使用 gVisor
# /etc/docker/daemon.json
{
  "runtimes": {
    "runsc": {
      "path": "/usr/bin/runsc"
    }
  }
}

# 使用 gVisor 运行容器
docker run --runtime=runsc nginx
```

### Kata Containers

使用轻量级虚拟机提供强隔离。

```bash
# 配置 Docker 使用 Kata
{
  "runtimes": {
    "kata": {
      "path": "/usr/bin/kata-runtime"
    }
  }
}

# 使用 Kata 运行容器
docker run --runtime=kata nginx
```

### 运行时对比

| 运行时 | 隔离级别 | 性能 | 兼容性 | 使用场景 |
|--------|----------|------|--------|----------|
| **runc** | 进程级 | 最高 | 最好 | 默认选择 |
| **gVisor** | 用户态内核 | 中等 | 良好 | 多租户环境 |
| **Kata** | 虚拟机级 | 较低 | 良好 | 高安全需求 |

## 调试运行时

```bash
# 查看 Docker 运行时信息
docker info | grep -i runtime

# 查看 containerd 状态
sudo systemctl status containerd
sudo ctr version

# 查看 containerd 日志
sudo journalctl -u containerd -f

# 调试容器创建过程
docker run --rm -it --runtime=runc \
  --log-driver=json-file \
  alpine sh
```
