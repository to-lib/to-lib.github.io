---
sidebar_position: 32
title: Docker Extensions
description: Docker Desktop 扩展开发与使用指南
---

# Docker Extensions

Docker Extensions 允许在 Docker Desktop 中添加第三方工具和功能。

## 使用扩展

### 浏览和安装

```bash
# 命令行安装
docker extension install docker/disk-usage-extension

# 或在 Docker Desktop 中：
# Extensions > Marketplace > 搜索并安装
```

### 常用扩展

| 扩展 | 功能 |
|------|------|
| **Disk Usage** | 可视化磁盘使用 |
| **Logs Explorer** | 日志查看和搜索 |
| **Portainer** | 容器管理 UI |
| **Snyk** | 安全扫描 |
| **Tailscale** | VPN 网络 |
| **LocalStack** | AWS 本地模拟 |
| **Grafana** | 监控仪表板 |

### 管理扩展

```bash
# 列出已安装扩展
docker extension ls

# 更新扩展
docker extension update docker/disk-usage-extension

# 卸载扩展
docker extension rm docker/disk-usage-extension
```

## 开发扩展

### 扩展结构

```
my-extension/
├── Dockerfile           # 扩展镜像
├── metadata.json        # 扩展元数据
├── docker-compose.yaml  # 可选：后端服务
└── ui/                  # 前端代码
    ├── package.json
    ├── src/
    └── build/
```

### metadata.json

```json
{
  "icon": "docker.svg",
  "ui": {
    "dashboard-tab": {
      "title": "My Extension",
      "root": "/ui",
      "src": "index.html"
    }
  },
  "vm": {
    "composefile": "docker-compose.yaml"
  },
  "host": {
    "binaries": [
      {
        "darwin": [{ "path": "/darwin/mybin" }],
        "linux": [{ "path": "/linux/mybin" }],
        "windows": [{ "path": "/windows/mybin.exe" }]
      }
    ]
  }
}
```

### Dockerfile

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY ui/package*.json ./
RUN npm ci
COPY ui/ .
RUN npm run build

FROM scratch
LABEL org.opencontainers.image.title="My Extension"
LABEL org.opencontainers.image.description="My awesome Docker extension"
LABEL org.opencontainers.image.vendor="My Company"
LABEL com.docker.desktop.extension.api.version="0.3.4"
LABEL com.docker.desktop.extension.icon="docker.svg"

COPY metadata.json .
COPY docker.svg .
COPY --from=builder /app/build ui
```

### 前端开发 (React)

```jsx
// ui/src/App.jsx
import { createDockerDesktopClient } from '@docker/extension-api-client';

const ddClient = createDockerDesktopClient();

function App() {
  const [containers, setContainers] = useState([]);

  useEffect(() => {
    // 调用 Docker API
    ddClient.docker.listContainers().then(setContainers);
  }, []);

  const runContainer = async () => {
    await ddClient.docker.cli.exec('run', ['-d', 'nginx']);
    // 刷新列表
    const updated = await ddClient.docker.listContainers();
    setContainers(updated);
  };

  return (
    <div>
      <h1>My Extension</h1>
      <button onClick={runContainer}>Run Nginx</button>
      <ul>
        {containers.map(c => (
          <li key={c.Id}>{c.Names[0]}: {c.State}</li>
        ))}
      </ul>
    </div>
  );
}
```

### Extension API

```javascript
const ddClient = createDockerDesktopClient();

// Docker CLI
await ddClient.docker.cli.exec('ps', ['-a']);
await ddClient.docker.cli.exec('build', ['-t', 'myapp', '.']);

// Docker API
const containers = await ddClient.docker.listContainers();
const images = await ddClient.docker.listImages();

// 执行主机命令
const result = await ddClient.extension.host.cli.exec('ls', ['-la']);

// 打开外部链接
ddClient.host.openExternal('https://example.com');

// 显示通知
ddClient.desktopUI.toast.success('Operation completed!');
ddClient.desktopUI.toast.error('Something went wrong');

// 导航
ddClient.desktopUI.navigate.viewContainers();
ddClient.desktopUI.navigate.viewImages();
```

### 后端服务

```yaml
# docker-compose.yaml
services:
  backend:
    build:
      context: backend
    ports:
      - 8080:8080
```

```javascript
// 前端调用后端
const response = await ddClient.extension.vm.service.get('/api/data');
const result = await ddClient.extension.vm.service.post('/api/action', { data: 'value' });
```

## 构建和测试

```bash
# 构建扩展
docker build -t my-extension .

# 安装本地扩展
docker extension install my-extension

# 开发模式（热重载）
docker extension dev ui-source my-extension ./ui
docker extension dev debug my-extension

# 重置开发模式
docker extension dev reset my-extension

# 查看日志
docker extension logs my-extension
```

## 发布扩展

```bash
# 推送到 Docker Hub
docker tag my-extension username/my-extension:1.0.0
docker push username/my-extension:1.0.0

# 提交到 Extensions Marketplace
# 访问 https://hub.docker.com/extensions
```

### 发布要求

- 有效的 metadata.json
- 清晰的图标和描述
- 遵循安全最佳实践
- 提供文档和支持渠道
