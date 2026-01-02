---
sidebar_position: 28
title: Docker API
description: Docker REST API 与 SDK 编程指南
---

# Docker API

Docker 提供 REST API 用于程序化管理容器、镜像、网络等资源。

## API 概述

Docker API 通过 Unix socket 或 TCP 端口暴露：

- 本地：`/var/run/docker.sock`
- 远程：`tcp://host:2376`（带 TLS）

### API 版本

```bash
# 查看 API 版本
docker version --format '{{.Server.APIVersion}}'

# API 端点带版本前缀
# /v1.44/containers/json
```

## 直接调用 API

### 使用 curl

```bash
# 通过 Unix socket
curl --unix-socket /var/run/docker.sock http://localhost/version

# 列出容器
curl --unix-socket /var/run/docker.sock http://localhost/containers/json

# 列出镜像
curl --unix-socket /var/run/docker.sock http://localhost/images/json

# 创建容器
curl --unix-socket /var/run/docker.sock \
  -H "Content-Type: application/json" \
  -d '{"Image": "nginx", "ExposedPorts": {"80/tcp": {}}}' \
  http://localhost/containers/create?name=mynginx

# 启动容器
curl --unix-socket /var/run/docker.sock \
  -X POST \
  http://localhost/containers/mynginx/start

# 停止容器
curl --unix-socket /var/run/docker.sock \
  -X POST \
  http://localhost/containers/mynginx/stop

# 删除容器
curl --unix-socket /var/run/docker.sock \
  -X DELETE \
  http://localhost/containers/mynginx
```

### 远程 API（带 TLS）

```bash
curl --cacert ca.pem --cert cert.pem --key key.pem \
  https://docker-host:2376/version
```

## 常用 API 端点

### 容器操作

| 方法   | 端点                      | 说明      |
| ------ | ------------------------- | --------- |
| GET    | `/containers/json`        | 列出容器  |
| POST   | `/containers/create`      | 创建容器  |
| POST   | `/containers/:id/start`   | 启动容器  |
| POST   | `/containers/:id/stop`    | 停止容器  |
| POST   | `/containers/:id/restart` | 重启容器  |
| DELETE | `/containers/:id`         | 删除容器  |
| GET    | `/containers/:id/logs`    | 获取日志  |
| GET    | `/containers/:id/stats`   | 获取统计  |
| POST   | `/containers/:id/exec`    | 创建 exec |

### 镜像操作

| 方法   | 端点                 | 说明     |
| ------ | -------------------- | -------- |
| GET    | `/images/json`       | 列出镜像 |
| POST   | `/images/create`     | 拉取镜像 |
| POST   | `/build`             | 构建镜像 |
| DELETE | `/images/:name`      | 删除镜像 |
| POST   | `/images/:name/push` | 推送镜像 |

## Python SDK

### 安装

```bash
pip install docker
```

### 基本使用

```python
import docker

# 连接本地 Docker
client = docker.from_env()

# 或指定连接
client = docker.DockerClient(base_url='unix://var/run/docker.sock')
# client = docker.DockerClient(base_url='tcp://192.168.1.100:2376')

# 查看版本
print(client.version())
```

### 容器操作

```python
import docker
client = docker.from_env()

# 运行容器
container = client.containers.run(
    "nginx",
    detach=True,
    ports={'80/tcp': 8080},
    name="my-nginx"
)

# 列出容器
for c in client.containers.list():
    print(f"{c.name}: {c.status}")

# 获取容器
container = client.containers.get("my-nginx")

# 容器操作
container.stop()
container.start()
container.restart()
container.remove()

# 执行命令
result = container.exec_run("ls -la")
print(result.output.decode())

# 查看日志
logs = container.logs()
print(logs.decode())

# 流式日志
for log in container.logs(stream=True):
    print(log.decode(), end='')
```

### 镜像操作

```python
# 拉取镜像
image = client.images.pull("nginx:latest")

# 列出镜像
for img in client.images.list():
    print(img.tags)

# 构建镜像
image, logs = client.images.build(
    path=".",
    tag="myapp:v1",
    rm=True
)
for log in logs:
    print(log)

# 推送镜像
client.images.push("myregistry/myapp:v1")
```

### 网络和卷

```python
# 创建网络
network = client.networks.create("mynet", driver="bridge")

# 创建卷
volume = client.volumes.create("mydata")

# 运行容器并连接
container = client.containers.run(
    "nginx",
    detach=True,
    network="mynet",
    volumes={"mydata": {"bind": "/data", "mode": "rw"}}
)
```

## Go SDK

### 安装

```bash
go get github.com/docker/docker/client
```

### 基本使用

```go
package main

import (
    "context"
    "fmt"
    "github.com/docker/docker/api/types"
    "github.com/docker/docker/api/types/container"
    "github.com/docker/docker/client"
)

func main() {
    ctx := context.Background()
    cli, err := client.NewClientWithOpts(client.FromEnv)
    if err != nil {
        panic(err)
    }
    defer cli.Close()

    // 列出容器
    containers, err := cli.ContainerList(ctx, container.ListOptions{})
    if err != nil {
        panic(err)
    }

    for _, c := range containers {
        fmt.Printf("%s: %s\n", c.Names[0], c.State)
    }
}
```

### 创建和运行容器

```go
package main

import (
    "context"
    "github.com/docker/docker/api/types/container"
    "github.com/docker/docker/api/types/image"
    "github.com/docker/docker/client"
    "github.com/docker/go-connections/nat"
    "io"
    "os"
)

func main() {
    ctx := context.Background()
    cli, _ := client.NewClientWithOpts(client.FromEnv)

    // 拉取镜像
    reader, _ := cli.ImagePull(ctx, "nginx:latest", image.PullOptions{})
    io.Copy(os.Stdout, reader)

    // 创建容器
    resp, _ := cli.ContainerCreate(ctx,
        &container.Config{
            Image: "nginx",
            ExposedPorts: nat.PortSet{
                "80/tcp": struct{}{},
            },
        },
        &container.HostConfig{
            PortBindings: nat.PortMap{
                "80/tcp": []nat.PortBinding{
                    {HostIP: "0.0.0.0", HostPort: "8080"},
                },
            },
        },
        nil, nil, "my-nginx")

    // 启动容器
    cli.ContainerStart(ctx, resp.ID, container.StartOptions{})
}
```

## Node.js SDK

### 安装

```bash
npm install dockerode
```

### 基本使用

```javascript
const Docker = require("dockerode");
const docker = new Docker({ socketPath: "/var/run/docker.sock" });

// 列出容器
async function listContainers() {
  const containers = await docker.listContainers();
  containers.forEach((c) => {
    console.log(`${c.Names[0]}: ${c.State}`);
  });
}

// 运行容器
async function runContainer() {
  const container = await docker.createContainer({
    Image: "nginx",
    name: "my-nginx",
    ExposedPorts: { "80/tcp": {} },
    HostConfig: {
      PortBindings: { "80/tcp": [{ HostPort: "8080" }] },
    },
  });
  await container.start();
  return container;
}

// 执行命令
async function execCommand(containerId, cmd) {
  const container = docker.getContainer(containerId);
  const exec = await container.exec({
    Cmd: cmd,
    AttachStdout: true,
    AttachStderr: true,
  });
  const stream = await exec.start();
  stream.pipe(process.stdout);
}
```

## 事件监听

### Python

```python
# 监听 Docker 事件
for event in client.events(decode=True):
    print(f"{event['Type']}: {event['Action']} - {event.get('Actor', {}).get('Attributes', {}).get('name', '')}")
```

### Go

```go
events, errs := cli.Events(ctx, types.EventsOptions{})
for {
    select {
    case event := <-events:
        fmt.Printf("%s: %s\n", event.Type, event.Action)
    case err := <-errs:
        fmt.Println(err)
        return
    }
}
```

## 安全注意事项

```bash
# 不要将 Docker socket 暴露给不信任的容器
# 这相当于给予 root 权限

# ❌ 危险
docker run -v /var/run/docker.sock:/var/run/docker.sock untrusted-image

# ✅ 使用 API 代理限制权限
# 如 Tecnativa/docker-socket-proxy
docker run -d \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e CONTAINERS=1 \
  -e IMAGES=1 \
  -p 2375:2375 \
  tecnativa/docker-socket-proxy
```
