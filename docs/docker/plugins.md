---
sidebar_position: 27
title: Docker Plugins
description: 网络、存储与授权插件的使用与开发
---

# Docker Plugins

Docker 插件系统允许扩展 Docker 的网络、存储和授权功能。

## 插件类型

| 类型 | 说明 | 示例 |
|------|------|------|
| **Volume** | 存储驱动插件 | NFS、GlusterFS、Ceph |
| **Network** | 网络驱动插件 | Weave、Calico |
| **Authorization** | 授权插件 | OPA、Casbin |
| **Log** | 日志驱动插件 | Splunk、Datadog |

## 插件管理

### 基本命令

```bash
# 搜索插件
docker plugin search volume

# 安装插件
docker plugin install vieux/sshfs

# 列出已安装插件
docker plugin ls

# 查看插件详情
docker plugin inspect vieux/sshfs

# 启用/禁用插件
docker plugin enable vieux/sshfs
docker plugin disable vieux/sshfs

# 删除插件
docker plugin rm vieux/sshfs
```

### 安装时配置

```bash
# 授予权限
docker plugin install vieux/sshfs --grant-all-permissions

# 设置配置项
docker plugin install vieux/sshfs DEBUG=1
```

## 存储插件

### SSHFS 插件

```bash
# 安装
docker plugin install vieux/sshfs

# 创建卷
docker volume create -d vieux/sshfs \
  -o sshcmd=user@host:/path \
  -o password=secret \
  sshvolume

# 使用卷
docker run -v sshvolume:/data alpine ls /data
```

### NFS 插件

```bash
# 使用本地驱动的 NFS 支持
docker volume create \
  --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.100,rw \
  --opt device=:/shared \
  nfs-vol
```

### REX-Ray 插件

```bash
# 安装 AWS EBS 插件
docker plugin install rexray/ebs \
  EBS_ACCESSKEY=xxx \
  EBS_SECRETKEY=xxx

# 创建 EBS 卷
docker volume create -d rexray/ebs --opt=size=100 ebs-vol
```

## 网络插件

### Weave Net

```bash
# 安装 Weave
curl -L git.io/weave -o /usr/local/bin/weave
chmod +x /usr/local/bin/weave
weave launch

# 创建 Weave 网络
docker network create --driver weave mynet

# 运行容器
docker run --network weave -d nginx
```

### Calico

```bash
# Calico 通常与 Kubernetes 一起使用
# 独立使用需要 etcd

# 创建 Calico 网络
docker network create --driver calico --ipam-driver calico-ipam calico-net
```

## 授权插件

### OPA (Open Policy Agent)

```bash
# 安装 OPA 插件
docker plugin install openpolicyagent/opa-docker-authz-v2:0.4

# 配置 Docker daemon
# /etc/docker/daemon.json
{
  "authorization-plugins": ["openpolicyagent/opa-docker-authz-v2:0.4"]
}
```

策略示例：

```rego
# policy.rego
package docker.authz

default allow = false

# 允许读取操作
allow {
    input.Method == "GET"
}

# 禁止特权容器
allow {
    input.Method == "POST"
    input.Path == "/containers/create"
    not input.Body.HostConfig.Privileged
}
```

## 开发自定义插件

### 插件结构

```
my-plugin/
├── config.json      # 插件配置
├── rootfs/          # 插件文件系统
│   └── plugin-binary
└── Dockerfile
```

### config.json 示例

```json
{
  "description": "My custom volume plugin",
  "documentation": "https://example.com/docs",
  "entrypoint": ["/plugin-binary"],
  "interface": {
    "types": ["docker.volumedriver/1.0"],
    "socket": "plugin.sock"
  },
  "network": {
    "type": "host"
  },
  "mounts": [
    {
      "source": "/var/lib/docker/plugins/",
      "destination": "/mnt/state",
      "type": "bind"
    }
  ],
  "env": [
    {
      "name": "DEBUG",
      "description": "Enable debug logging",
      "value": "0"
    }
  ]
}
```

### Volume 插件接口

```go
// Go 实现示例
package main

import (
    "github.com/docker/go-plugins-helpers/volume"
)

type MyDriver struct {
    volumes map[string]string
}

func (d *MyDriver) Create(req *volume.CreateRequest) error {
    d.volumes[req.Name] = "/data/" + req.Name
    return nil
}

func (d *MyDriver) Remove(req *volume.RemoveRequest) error {
    delete(d.volumes, req.Name)
    return nil
}

func (d *MyDriver) Mount(req *volume.MountRequest) (*volume.MountResponse, error) {
    return &volume.MountResponse{Mountpoint: d.volumes[req.Name]}, nil
}

func (d *MyDriver) Unmount(req *volume.UnmountRequest) error {
    return nil
}

func (d *MyDriver) Path(req *volume.PathRequest) (*volume.PathResponse, error) {
    return &volume.PathResponse{Mountpoint: d.volumes[req.Name]}, nil
}

func (d *MyDriver) Get(req *volume.GetRequest) (*volume.GetResponse, error) {
    return &volume.GetResponse{Volume: &volume.Volume{Name: req.Name}}, nil
}

func (d *MyDriver) List() (*volume.ListResponse, error) {
    var vols []*volume.Volume
    for name := range d.volumes {
        vols = append(vols, &volume.Volume{Name: name})
    }
    return &volume.ListResponse{Volumes: vols}, nil
}

func (d *MyDriver) Capabilities() *volume.CapabilitiesResponse {
    return &volume.CapabilitiesResponse{Capabilities: volume.Capability{Scope: "local"}}
}

func main() {
    driver := &MyDriver{volumes: make(map[string]string)}
    handler := volume.NewHandler(driver)
    handler.ServeUnix("my-plugin", 0)
}
```

### 构建和安装插件

```bash
# 构建插件
docker build -t my-plugin .

# 创建插件
docker plugin create my-plugin ./my-plugin

# 启用插件
docker plugin enable my-plugin

# 测试
docker volume create -d my-plugin test-vol
```
