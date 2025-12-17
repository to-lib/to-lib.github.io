---
sidebar_position: 12
title: 快速参考
description: Podman 命令快速参考
---

# Podman 快速参考

## 容器生命周期

| 命令                             | 说明           |
| -------------------------------- | -------------- |
| `podman run -d -p 80:80 nginx`   | 运行容器       |
| `podman run -it alpine sh`       | 交互式运行     |
| `podman ps -a`                   | 列出所有容器   |
| `podman start/stop/restart <id>` | 启动/停止/重启 |
| `podman rm <id>`                 | 删除容器       |
| `podman rm -f $(podman ps -aq)`  | 删除所有容器   |

## 容器运行

```bash
podman run -d \
  --name myapp \
  -p 8080:80 \
  -v /data:/data \
  -e KEY=value \
  --network mynet \
  --memory=512m \
  --cpus=0.5 \
  nginx
```

## 镜像管理

| 命令                           | 说明     |
| ------------------------------ | -------- |
| `podman images`                | 列出镜像 |
| `podman pull nginx`            | 拉取镜像 |
| `podman build -t app .`        | 构建镜像 |
| `podman push app registry/app` | 推送镜像 |
| `podman save -o app.tar app`   | 保存镜像 |
| `podman load -i app.tar`       | 加载镜像 |
| `podman rmi <id>`              | 删除镜像 |

## Pod 管理

```bash
# 创建 Pod
podman pod create --name mypod -p 8080:80

# 在 Pod 中运行
podman run -d --pod mypod nginx

# 列出/管理 Pod
podman pod ls
podman pod start/stop mypod
podman pod rm mypod
```

## 网络管理

| 命令                                   | 说明     |
| -------------------------------------- | -------- |
| `podman network create mynet`          | 创建网络 |
| `podman network ls`                    | 列出网络 |
| `podman network connect mynet <id>`    | 连接网络 |
| `podman network disconnect mynet <id>` | 断开网络 |
| `podman network rm mynet`              | 删除网络 |

## 卷管理

| 命令                           | 说明       |
| ------------------------------ | ---------- |
| `podman volume create mydata`  | 创建卷     |
| `podman volume ls`             | 列出卷     |
| `podman volume inspect mydata` | 查看详情   |
| `podman volume rm mydata`      | 删除卷     |
| `podman volume prune`          | 清理未使用 |

## Systemd 集成

```bash
# 生成服务
podman generate systemd --new --name web > web.service

# 用户服务
systemctl --user enable --now web
loginctl enable-linger $USER

# Quadlet (Podman 4.4+)
# 创建 ~/.config/containers/systemd/web.container
```

## Rootless 配置

```bash
# 配置 subuid/subgid
sudo usermod --add-subuids 100000-165535 $USER
sudo usermod --add-subgids 100000-165535 $USER
podman system migrate
```

## 调试命令

| 命令                      | 说明     |
| ------------------------- | -------- |
| `podman exec -it <id> sh` | 进入容器 |
| `podman logs -f <id>`     | 查看日志 |
| `podman top <id>`         | 查看进程 |
| `podman stats`            | 资源使用 |
| `podman inspect <id>`     | 详细信息 |
| `podman port <id>`        | 端口映射 |

## 系统管理

```bash
podman system prune -a     # 清理所有
podman system df           # 查看空间
podman system reset        # 重置 Podman
podman info                # 系统信息
```

## Buildah 常用

```bash
buildah from alpine        # 创建容器
buildah run $ctr cmd       # 运行命令
buildah copy $ctr src dst  # 复制文件
buildah commit $ctr img    # 提交镜像
```

## Skopeo 常用

```bash
skopeo inspect docker://nginx        # 检查镜像
skopeo copy docker://a docker://b    # 复制镜像
skopeo list-tags docker://nginx      # 列出标签
```

## Docker 兼容

```bash
alias docker=podman
# 或
sudo dnf install podman-docker

# Socket 兼容
systemctl --user enable --now podman.socket
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock
```

## 安全运行

```bash
podman run -d \
  --read-only \
  --cap-drop=all \
  --security-opt=no-new-privileges \
  --user 1000:1000 \
  nginx
```

## Kubernetes 集成

```bash
# 生成 YAML
podman generate kube mypod > pod.yaml

# 从 YAML 运行
podman play kube pod.yaml

# 删除
podman play kube --down pod.yaml
```
