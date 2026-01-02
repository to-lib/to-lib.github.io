---
sidebar_position: 25
title: 开发者体验
description: Dev Containers、Docker Desktop 替代品与 Testcontainers
---

# 开发者体验

提升 Docker 开发体验的工具和最佳实践。

## Dev Containers

VS Code Dev Containers 允许在容器中进行开发，确保开发环境一致性。

### 基础配置

```json
// .devcontainer/devcontainer.json
{
  "name": "Node.js Development",
  "image": "mcr.microsoft.com/devcontainers/javascript-node:18",
  "features": {
    "ghcr.io/devcontainers/features/docker-in-docker:2": {}
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode"
      ],
      "settings": {
        "editor.formatOnSave": true
      }
    }
  },
  "forwardPorts": [3000],
  "postCreateCommand": "npm install",
  "remoteUser": "node"
}
```

### 使用 Dockerfile

```json
// .devcontainer/devcontainer.json
{
  "name": "Custom Dev Container",
  "build": {
    "dockerfile": "Dockerfile",
    "context": ".."
  },
  "mounts": [
    "source=${localWorkspaceFolder},target=/workspace,type=bind"
  ],
  "workspaceFolder": "/workspace"
}
```

```dockerfile
# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/base:ubuntu

RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 安装 Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

USER vscode
```

### 使用 Docker Compose

```json
// .devcontainer/devcontainer.json
{
  "name": "Full Stack Dev",
  "dockerComposeFile": "docker-compose.yml",
  "service": "app",
  "workspaceFolder": "/workspace",
  "shutdownAction": "stopCompose"
}
```

```yaml
# .devcontainer/docker-compose.yml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ..:/workspace:cached
    command: sleep infinity

  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: postgres

  redis:
    image: redis:alpine
```

## Docker Desktop 替代品

### Colima (macOS/Linux)

```bash
# 安装
brew install colima docker

# 启动（默认配置）
colima start

# 自定义资源
colima start --cpu 4 --memory 8 --disk 100

# 使用 containerd 运行时
colima start --runtime containerd

# 启用 Kubernetes
colima start --kubernetes

# 查看状态
colima status

# 停止
colima stop
```

### Rancher Desktop (跨平台)

```bash
# macOS 安装
brew install --cask rancher

# 特点：
# - 支持 containerd 和 dockerd
# - 内置 Kubernetes
# - 图形界面管理
# - 支持 Windows/macOS/Linux
```

### OrbStack (macOS)

```bash
# 安装
brew install --cask orbstack

# 特点：
# - 极快的启动速度
# - 低资源占用
# - 原生 macOS 集成
# - 支持 Linux 虚拟机
```

### Podman (跨平台)

```bash
# macOS 安装
brew install podman

# 初始化虚拟机
podman machine init
podman machine start

# 使用（命令兼容 Docker）
podman run -d -p 80:80 nginx
podman ps
podman images

# 设置别名
alias docker=podman
```

### 对比

| 工具 | 平台 | 资源占用 | Kubernetes | 特点 |
|------|------|----------|------------|------|
| Docker Desktop | 全平台 | 高 | 支持 | 官方，功能全面 |
| Colima | macOS/Linux | 低 | 支持 | 轻量，命令行 |
| Rancher Desktop | 全平台 | 中 | 支持 | GUI，多运行时 |
| OrbStack | macOS | 极低 | 支持 | 最快，原生集成 |
| Podman | 全平台 | 低 | 不支持 | 无守护进程 |

## Testcontainers

在集成测试中使用真实的容器化依赖。

### Java (JUnit 5)

```java
@Testcontainers
class UserServiceTest {

    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15")
        .withDatabaseName("testdb")
        .withUsername("test")
        .withPassword("test");

    @Container
    static GenericContainer<?> redis = new GenericContainer<>("redis:alpine")
        .withExposedPorts(6379);

    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", postgres::getJdbcUrl);
        registry.add("spring.datasource.username", postgres::getUsername);
        registry.add("spring.datasource.password", postgres::getPassword);
        registry.add("spring.redis.host", redis::getHost);
        registry.add("spring.redis.port", () -> redis.getMappedPort(6379));
    }

    @Test
    void testUserCreation() {
        // 测试代码
    }
}
```

### Node.js (Jest)

```javascript
const { GenericContainer, Wait } = require("testcontainers");

describe("User Service", () => {
  let postgresContainer;
  let redisContainer;

  beforeAll(async () => {
    postgresContainer = await new GenericContainer("postgres:15")
      .withEnvironment({
        POSTGRES_DB: "testdb",
        POSTGRES_USER: "test",
        POSTGRES_PASSWORD: "test"
      })
      .withExposedPorts(5432)
      .withWaitStrategy(Wait.forLogMessage(/ready to accept connections/))
      .start();

    redisContainer = await new GenericContainer("redis:alpine")
      .withExposedPorts(6379)
      .start();

    process.env.DATABASE_URL = `postgresql://test:test@${postgresContainer.getHost()}:${postgresContainer.getMappedPort(5432)}/testdb`;
    process.env.REDIS_URL = `redis://${redisContainer.getHost()}:${redisContainer.getMappedPort(6379)}`;
  }, 60000);

  afterAll(async () => {
    await postgresContainer?.stop();
    await redisContainer?.stop();
  });

  test("should create user", async () => {
    // 测试代码
  });
});
```

### Go

```go
package main

import (
    "context"
    "testing"
    "github.com/testcontainers/testcontainers-go"
    "github.com/testcontainers/testcontainers-go/wait"
)

func TestUserService(t *testing.T) {
    ctx := context.Background()

    // 启动 PostgreSQL 容器
    postgresC, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
        ContainerRequest: testcontainers.ContainerRequest{
            Image:        "postgres:15",
            ExposedPorts: []string{"5432/tcp"},
            Env: map[string]string{
                "POSTGRES_DB":       "testdb",
                "POSTGRES_USER":     "test",
                "POSTGRES_PASSWORD": "test",
            },
            WaitingFor: wait.ForLog("ready to accept connections"),
        },
        Started: true,
    })
    if err != nil {
        t.Fatal(err)
    }
    defer postgresC.Terminate(ctx)

    host, _ := postgresC.Host(ctx)
    port, _ := postgresC.MappedPort(ctx, "5432")

    // 使用 host 和 port 连接数据库进行测试
}
```

### Python (pytest)

```python
import pytest
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

@pytest.fixture(scope="session")
def postgres():
    with PostgresContainer("postgres:15") as postgres:
        yield postgres

@pytest.fixture(scope="session")
def redis():
    with RedisContainer("redis:alpine") as redis:
        yield redis

def test_user_creation(postgres, redis):
    # 获取连接信息
    db_url = postgres.get_connection_url()
    redis_host = redis.get_container_host_ip()
    redis_port = redis.get_exposed_port(6379)

    # 测试代码
    assert True
```

## 本地开发最佳实践

### 开发环境 Docker Compose

```yaml
# docker-compose.dev.yml
services:
  app:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - node_modules:/app/node_modules
    ports:
      - "3000:3000"
      - "9229:9229"  # 调试端口
    environment:
      - NODE_ENV=development
    command: npm run dev

  db:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: dev
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  mailhog:
    image: mailhog/mailhog
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web UI

volumes:
  node_modules:
  postgres-data:
```

### 热重载配置

```dockerfile
# Dockerfile
FROM node:18-alpine AS development
WORKDIR /app
COPY package*.json ./
RUN npm install
# 不复制源代码，通过 volume 挂载
CMD ["npm", "run", "dev"]

FROM node:18-alpine AS production
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
CMD ["npm", "start"]
```

### Makefile 简化命令

```makefile
.PHONY: dev build test clean

dev:
	docker compose -f docker-compose.dev.yml up

build:
	docker compose build

test:
	docker compose -f docker-compose.test.yml up --abort-on-container-exit

clean:
	docker compose down -v
	docker system prune -f

logs:
	docker compose logs -f

shell:
	docker compose exec app sh
```
