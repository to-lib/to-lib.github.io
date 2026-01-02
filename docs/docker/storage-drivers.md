---
sidebar_position: 21
title: 存储进阶
description: 存储驱动对比、tmpfs 挂载与分布式存储方案
---

# 存储进阶

深入理解 Docker 存储机制和生产环境存储方案。

## 存储驱动对比

### 主要存储驱动

| 驱动 | 后端文件系统 | 性能 | 稳定性 | 推荐场景 |
|------|-------------|------|--------|----------|
| **overlay2** | xfs, ext4 | 高 | 高 | 默认推荐 |
| **btrfs** | btrfs | 高 | 中 | 使用 btrfs 的系统 |
| **zfs** | zfs | 高 | 高 | 使用 zfs 的系统 |
| **devicemapper** | direct-lvm | 中 | 高 | RHEL/CentOS 旧版本 |
| **vfs** | 任意 | 低 | 高 | 测试/调试 |

### 查看当前存储驱动

```bash
docker info | grep "Storage Driver"
# Storage Driver: overlay2
```

### 配置存储驱动

```json
// /etc/docker/daemon.json
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ]
}
```

### overlay2 详解

```bash
# 查看 overlay2 存储位置
ls /var/lib/docker/overlay2/

# 查看容器的 overlay 挂载信息
docker inspect container --format '{{json .GraphDriver.Data}}' | jq

# 目录结构
# /var/lib/docker/overlay2/
# ├── <layer-id>/
# │   ├── diff/      # 该层的文件内容
# │   ├── link       # 短 ID 链接
# │   ├── lower      # 下层引用
# │   └── work/      # overlay 工作目录
# └── l/             # 短 ID 符号链接目录
```

## tmpfs 挂载

tmpfs 将数据存储在内存中，容器停止后数据消失。

### 使用场景

- 敏感数据（不写入磁盘）
- 高性能临时存储
- 缓存数据

### 基本用法

```bash
# 挂载 tmpfs
docker run -d \
  --tmpfs /tmp \
  --tmpfs /run:rw,noexec,nosuid,size=100m \
  nginx

# 使用 --mount 语法
docker run -d \
  --mount type=tmpfs,destination=/tmp,tmpfs-size=100m,tmpfs-mode=1777 \
  nginx
```

### tmpfs 选项

| 选项 | 说明 | 示例 |
|------|------|------|
| `size` | 大小限制 | `size=100m` |
| `mode` | 文件权限 | `mode=1777` |
| `uid` | 所有者 UID | `uid=1000` |
| `gid` | 所有者 GID | `gid=1000` |

### Docker Compose tmpfs

```yaml
services:
  app:
    image: myapp
    tmpfs:
      - /tmp
      - /run:size=100m,mode=1777
```

## 绑定挂载 vs 卷

### 对比

| 特性 | 绑定挂载 (Bind Mount) | 卷 (Volume) |
|------|----------------------|-------------|
| 位置 | 主机任意路径 | /var/lib/docker/volumes |
| 管理 | 手动管理 | Docker 管理 |
| 备份 | 直接备份主机目录 | 需要通过容器或 API |
| 性能 | 取决于主机文件系统 | 优化过的性能 |
| 可移植性 | 依赖主机路径 | 更好的可移植性 |

### 绑定挂载

```bash
# 基本绑定挂载
docker run -v /host/path:/container/path nginx

# 只读挂载
docker run -v /host/path:/container/path:ro nginx

# 使用 --mount 语法（推荐）
docker run --mount type=bind,source=/host/path,target=/container/path nginx
```

### 命名卷

```bash
# 创建卷
docker volume create mydata

# 使用卷
docker run -v mydata:/data nginx

# 查看卷详情
docker volume inspect mydata
```

## NFS 存储

### 创建 NFS 卷

```bash
# 创建 NFS 卷
docker volume create \
  --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.100,rw,nfsvers=4 \
  --opt device=:/path/to/share \
  nfs-data

# 使用 NFS 卷
docker run -v nfs-data:/data nginx
```

### Docker Compose NFS

```yaml
services:
  app:
    image: myapp
    volumes:
      - nfs-data:/data

volumes:
  nfs-data:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.100,rw,nfsvers=4
      device: ":/path/to/share"
```

## 分布式存储

### Ceph RBD

```bash
# 安装 Ceph 卷插件
docker plugin install rexray/ceph

# 创建 Ceph 卷
docker volume create -d rexray/ceph myvolume

# 使用卷
docker run -v myvolume:/data nginx
```

### GlusterFS

```yaml
# Docker Compose with GlusterFS
volumes:
  gluster-data:
    driver: local
    driver_opts:
      type: glusterfs
      o: addr=gluster-server
      device: ":/gv0"
```

### 云存储

```bash
# AWS EBS (使用 REX-Ray)
docker plugin install rexray/ebs

# Azure Disk
docker plugin install rexray/azureud

# Google Cloud Persistent Disk
docker plugin install rexray/gcepd
```

## 存储性能优化

### 选择合适的存储驱动

```bash
# 检查文件系统类型
df -T /var/lib/docker

# overlay2 推荐使用 xfs 或 ext4
# 如果使用 xfs，确保启用 d_type
xfs_info /var/lib/docker | grep ftype
# ftype=1 表示支持
```

### 数据目录分离

```json
// /etc/docker/daemon.json
{
  "data-root": "/mnt/docker-data"
}
```

### 日志存储优化

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

## 数据备份与恢复

### 备份卷数据

```bash
# 使用临时容器备份
docker run --rm \
  -v mydata:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/mydata-backup.tar.gz -C /data .

# 恢复数据
docker run --rm \
  -v mydata:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/mydata-backup.tar.gz -C /data
```

### 备份容器

```bash
# 导出容器文件系统
docker export container > container-backup.tar

# 导入为镜像
docker import container-backup.tar myimage:backup
```

### 迁移卷到新主机

```bash
# 源主机：导出卷
docker run --rm -v mydata:/data -v /tmp:/backup alpine \
  tar czf /backup/mydata.tar.gz -C /data .

# 传输文件
scp /tmp/mydata.tar.gz user@newhost:/tmp/

# 目标主机：创建卷并导入
docker volume create mydata
docker run --rm -v mydata:/data -v /tmp:/backup alpine \
  tar xzf /backup/mydata.tar.gz -C /data
```
