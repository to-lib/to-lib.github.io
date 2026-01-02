---
sidebar_position: 33
title: Windows Containers
description: Windows 容器基础与使用指南
---

# Windows Containers

Windows 容器允许在 Windows 上运行原生 Windows 应用程序，与 Linux 容器不同。

## 容器类型

### Windows Server Containers

- 与主机共享内核
- 轻量级，启动快
- 适合同版本 Windows

### Hyper-V Containers

- 每个容器运行在轻量级 VM 中
- 更强的隔离性
- 可运行不同版本 Windows

## 环境准备

### 启用 Windows 容器功能

```powershell
# Windows Server
Install-WindowsFeature -Name Containers

# Windows 10/11 (需要 Pro/Enterprise)
Enable-WindowsOptionalFeature -Online -FeatureName Containers -All
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
```

### Docker Desktop 配置

```
Settings > General > Use the WSL 2 based engine (关闭)
Settings > General > Switch to Windows containers
```

或右键托盘图标 > "Switch to Windows containers"

## 基础镜像

### 官方基础镜像

| 镜像 | 大小 | 用途 |
|------|------|------|
| `mcr.microsoft.com/windows/servercore` | ~5GB | 完整 Windows Server Core |
| `mcr.microsoft.com/windows/nanoserver` | ~100MB | 最小化 Windows |
| `mcr.microsoft.com/windows` | ~10GB | 完整 Windows |
| `mcr.microsoft.com/dotnet/aspnet` | ~300MB | ASP.NET 运行时 |

### 版本匹配

```powershell
# 查看主机版本
[System.Environment]::OSVersion.Version

# 镜像标签需要匹配主机版本
# ltsc2022 - Windows Server 2022
# ltsc2019 - Windows Server 2019
# 1809, 1903, 1909, 2004, 20H2 - 半年频道版本
```

## Dockerfile 示例

### .NET Framework 应用

```dockerfile
FROM mcr.microsoft.com/dotnet/framework/aspnet:4.8-windowsservercore-ltsc2022

WORKDIR /inetpub/wwwroot
COPY ./publish .

EXPOSE 80
```

### .NET Core/6+ 应用

```dockerfile
FROM mcr.microsoft.com/dotnet/sdk:8.0-nanoserver-ltsc2022 AS build
WORKDIR /src
COPY *.csproj .
RUN dotnet restore
COPY . .
RUN dotnet publish -c Release -o /app

FROM mcr.microsoft.com/dotnet/aspnet:8.0-nanoserver-ltsc2022
WORKDIR /app
COPY --from=build /app .
ENTRYPOINT ["dotnet", "MyApp.dll"]
```

### IIS 应用

```dockerfile
FROM mcr.microsoft.com/windows/servercore/iis:windowsservercore-ltsc2022

RUN powershell -Command \
    Remove-Item -Recurse C:\inetpub\wwwroot\*

COPY ./website C:/inetpub/wwwroot

EXPOSE 80
```

## 运行容器

```powershell
# 运行 Windows 容器
docker run -d -p 8080:80 mcr.microsoft.com/windows/servercore/iis

# 使用 Hyper-V 隔离
docker run -d --isolation=hyperv -p 8080:80 myapp

# 进入容器
docker exec -it container_id powershell
```

## 网络配置

### 网络模式

| 模式 | 说明 |
|------|------|
| nat | 默认，NAT 网络 |
| transparent | 直接连接物理网络 |
| overlay | Swarm 跨主机网络 |
| l2bridge | L2 桥接 |

```powershell
# 创建 NAT 网络
docker network create -d nat mynat

# 创建透明网络
docker network create -d transparent mytrans
```

## 存储

```powershell
# 创建卷
docker volume create mydata

# 挂载卷
docker run -v mydata:C:\data myapp

# 绑定挂载
docker run -v C:\HostPath:C:\ContainerPath myapp
```

## Docker Compose

```yaml
version: '3.8'

services:
  web:
    image: mcr.microsoft.com/dotnet/aspnet:8.0-nanoserver-ltsc2022
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:80"
    volumes:
      - app-data:C:\app\data

  db:
    image: mcr.microsoft.com/mssql/server:2022-latest
    environment:
      SA_PASSWORD: "YourStrong!Passw0rd"
      ACCEPT_EULA: "Y"
    ports:
      - "1433:1433"

volumes:
  app-data:
```

## 常见问题

### 版本不匹配

```
container operating system does not match host operating system
```

解决：使用匹配主机版本的镜像标签，或使用 Hyper-V 隔离。

### 镜像太大

- 使用 nanoserver 替代 servercore
- 使用多阶段构建
- 清理临时文件

### 性能优化

```powershell
# 使用进程隔离（更快）
docker run --isolation=process myapp

# 预拉取基础镜像
docker pull mcr.microsoft.com/windows/nanoserver:ltsc2022
```

## Linux vs Windows 容器

| 特性 | Linux | Windows |
|------|-------|---------|
| 镜像大小 | MB 级 | GB 级 |
| 启动速度 | 秒级 | 较慢 |
| 基础镜像 | Alpine, Debian | Nano, ServerCore |
| 生态系统 | 丰富 | 有限 |
| 适用场景 | 大多数应用 | .NET Framework, IIS |
