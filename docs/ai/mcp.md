---
sidebar_position: 3
title: 🔌 MCP (模型上下文协议)
---

# MCP (Model Context Protocol)

**Model Context Protocol (MCP)** 是一个开放的标准协议，旨在解决 AI 模型与外部数据和工具连接的“最后一公里”问题。它由 Anthropic 等公司推动，致力于提供一种通用的方式，让 AI 助手能够安全、一致地访问本地和远程资源。

## 背景与问题

目前，将 AI 连接到数据源（如数据库、API、本地文件）通常需要为每个数据源编写特定的“连接器”或“插件”。这导致了：

- **碎片化**：每个 AI 平台都有自己的插件标准。
- **重复造轮子**：开发者需要为不同的 AI 平台重复开发相同的连接器。
- **维护困难**：数据源 API 变更需要更新所有相关的连接器。

## MCP 的解决方案

MCP 采用 **Client-Host-Server** 架构：

- **MCP Host (主机)**：运行 AI 模型的应用程序（如 Claude Desktop, IDE）。
- **MCP Server (服务器)**：提供数据或工具的独立服务（如 Google Drive Connector, Postgres Connector）。
- **MCP Client (客户端)**：Host 内部用于与 Server 通信的组件。

### 核心优势

1.  **通用性**：编写一次 MCP Server，即可在所有支持 MCP 的 Host（如 Claude Desktop, Zed Editor 等）中使用。
2.  **安全性**：用户可以精细控制 AI 对数据的访问权限。
3.  **标准化**：统一了资源 (Resources)、提示词 (Prompts) 和工具 (Tools) 的定义方式。

## 核心概念

### 1. Resources (资源)

数据源。AI 可以读取的内容。类似于 GET 请求。

- 文件内容
- 数据库记录
- API 返回的日志

### 2. Tools (工具)

可执行的操作。AI 可以调用的函数。类似于 POST 请求。

- 执行 SQL 查询
- 发送 API 请求
- 修改文件

### 3. Prompts (提示词)

预定义的提示词模板，用于引导 AI 完成特定任务。

## 如何开始

### 安装 MCP Server

通常可以通过 `npm` 或 `docker` 运行 MCP Server。例如：

```bash
npx -y @modelcontextprotocol/server-filesystem /path/to/directory
```

### 配置 Host

在 Claude Desktop 或其他支持 MCP 的应用配置文件中添加 Server 信息。

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/Desktop"
      ]
    }
  }
}
```

## 参考资源

- [MCP 官方文档](https://modelcontextprotocol.io/introduction)
- [MCP GitHub 仓库](https://github.com/modelcontextprotocol)
