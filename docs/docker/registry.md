---
sidebar_position: 6
title: 镜像仓库与发布
description: Docker 镜像命名、tag 策略、推送/拉取与私有仓库搭建
---

# 镜像仓库与发布

## 镜像命名与 Tag

镜像完整名称：

```text
[registry-host[:port]/][namespace/]repo[:tag]
```

示例：

- `nginx:1.25-alpine`
- `ghcr.io/org/app:1.2.3`
- `registry.example.com/team/app:2025-12-20`

建议的 tag 策略（从可回滚与可追溯角度）：

- `v1.2.3`（语义化版本）
- `v1.2.3-<gitsha>`（版本 + 提交）
- `sha-<gitsha>`（CI 自动）
- `latest` 仅用于“默认稳定版”（可用但不要依赖它做回滚）

## 登录、打标签、推送

```bash
# 登录仓库
docker login registry.example.com

# 构建镜像（本地 tag）
docker build -t app:local .

# 打标签（推送到远端仓库命名空间）
docker tag app:local registry.example.com/team/app:v1.0.0

# 推送
docker push registry.example.com/team/app:v1.0.0

# 拉取
docker pull registry.example.com/team/app:v1.0.0
```

## 选择仓库：Docker Hub / GHCR / Harbor

- **Docker Hub**：生态最大，公共镜像多
- **GHCR（GitHub Container Registry）**：与 GitHub Actions 配合方便，权限管理清晰
- **Harbor**：企业私有化常用，支持项目/镜像扫描/复制策略/审计等

## 自建 registry:2（最简私有仓库）

### 快速启动

```bash
docker run -d \
  --name registry \
  -p 5000:5000 \
  -v registry-data:/var/lib/registry \
  registry:2
```

推送/拉取：

```bash
docker tag app:local localhost:5000/app:v1

docker push localhost:5000/app:v1

docker pull localhost:5000/app:v1
```

### 配置为不安全仓库（仅测试环境）

`/etc/docker/daemon.json`：

```json
{
  "insecure-registries": ["localhost:5000"]
}
```

重启 Docker 后生效。

## TLS（生产建议）

生产环境建议为仓库配置 TLS（并配置证书信任链），常见做法：

- Nginx/Traefik 作为反向代理，终止 TLS
- 仓库域名使用受信 CA 证书（避免在客户端配 `insecure-registries`）

## 基础认证（htpasswd）

```bash
# 生成密码文件（需要 htpasswd 工具，通常来自 apache2-utils/httpd-tools）
htpasswd -Bbn user strong_password > htpasswd

# 启动带认证的 registry
docker run -d \
  --name registry \
  -p 5000:5000 \
  -v $(pwd)/htpasswd:/auth/htpasswd:ro \
  -e "REGISTRY_AUTH=htpasswd" \
  -e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
  -e "REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd" \
  -v registry-data:/var/lib/registry \
  registry:2
```

客户端登录：

```bash
docker login localhost:5000
```

## 镜像清理与空间治理

高频痛点：tag 很多但空间不释放。

- 镜像推送会产生历史 layer
- 删除 tag 不一定立即回收存储（与 registry 的垃圾回收策略有关）

registry:2 的垃圾回收通常需要：

- 配置 `storage.delete.enabled=true`
- 执行 `registry garbage-collect`（常在停机窗口操作）

企业场景更建议使用 Harbor 这类带生命周期管理的仓库。

## 与 CI/CD 集成（最小示例）

常见发布流程：

```bash
# 生成版本信息
VERSION=$(git describe --tags --always)
SHA=$(git rev-parse --short HEAD)

# 构建并打多个 tag
DOCKER_BUILDKIT=1 docker build -t app:${VERSION} -t app:${VERSION}-${SHA} .

# 推送到远端
REMOTE=registry.example.com/team/app

docker tag app:${VERSION} ${REMOTE}:${VERSION}
docker tag app:${VERSION}-${SHA} ${REMOTE}:${VERSION}-${SHA}

docker push ${REMOTE}:${VERSION}
docker push ${REMOTE}:${VERSION}-${SHA}
```
