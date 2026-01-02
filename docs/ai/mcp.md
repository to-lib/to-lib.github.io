---
sidebar_position: 8
title: 🔌 MCP (模型上下文协议)
---

# MCP (Model Context Protocol)

**Model Context Protocol (MCP)** 是一个开放的标准协议，旨在解决 AI 模型与外部数据和工具连接的"最后一公里"问题。它由 Anthropic 推动，致力于提供一种通用的方式，让 AI 助手能够安全、一致地访问本地和远程资源。

## 为什么需要 MCP？

目前，将 AI 连接到数据源（如数据库、API、本地文件）通常需要为每个数据源编写特定的"连接器"或"插件"。这导致了：

| 问题         | 说明                                           |
| ------------ | ---------------------------------------------- |
| **碎片化**   | 每个 AI 平台都有自己的插件标准                 |
| **重复造轮** | 开发者需要为不同的 AI 平台重复开发相同的连接器 |
| **维护困难** | 数据源 API 变更需要更新所有相关的连接器        |

MCP 通过标准化协议解决了这些问题。

## MCP 架构

MCP 采用 **Client-Host-Server** 架构：

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Host                             │
│  (Claude Desktop / IDE / AI Application)                │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ MCP Client  │  │ MCP Client  │  │ MCP Client  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
└─────────┼────────────────┼────────────────┼─────────────┘
          │                │                │
          ▼                ▼                ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │ MCP Server   │ │ MCP Server   │ │ MCP Server   │
   │ (Filesystem) │ │ (Database)   │ │ (Web API)    │
   └──────────────┘ └──────────────┘ └──────────────┘
```

- **MCP Host**：运行 AI 模型的应用程序（如 Claude Desktop, Cursor, Kiro）
- **MCP Client**：Host 内部用于与 Server 通信的组件
- **MCP Server**：提供数据或工具的独立服务

### 核心优势

1. **通用性**：编写一次 MCP Server，即可在所有支持 MCP 的 Host 中使用
2. **安全性**：用户可以精细控制 AI 对数据的访问权限
3. **标准化**：统一了资源、提示词和工具的定义方式
4. **可组合**：多个 Server 可以同时连接，能力可叠加

## 核心概念

### 1. Resources (资源)

数据源，AI 可以读取的内容。类似于 GET 请求。

```typescript
// 资源定义示例
{
  uri: "file:///path/to/document.md",
  name: "Project README",
  mimeType: "text/markdown"
}
```

常见资源类型：
- 文件内容
- 数据库记录
- API 返回的数据
- 实时日志

### 2. Tools (工具)

可执行的操作，AI 可以调用的函数。类似于 POST 请求。

```typescript
// 工具定义示例
{
  name: "query_database",
  description: "执行 SQL 查询",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "SQL 查询语句" }
    },
    required: ["query"]
  }
}
```

常见工具类型：
- 执行 SQL 查询
- 发送 HTTP 请求
- 读写文件
- 调用外部 API

### 3. Prompts (提示词模板)

预定义的提示词模板，用于引导 AI 完成特定任务。

```typescript
// 提示词模板示例
{
  name: "code_review",
  description: "代码审查模板",
  arguments: [
    { name: "code", description: "要审查的代码", required: true }
  ]
}
```

## 快速开始

### 使用现有 MCP Server

#### 1. 文件系统 Server

```bash
npx -y @modelcontextprotocol/server-filesystem /path/to/directory
```

#### 2. PostgreSQL Server

```bash
npx -y @modelcontextprotocol/server-postgres postgresql://user:pass@localhost/db
```

#### 3. GitHub Server

```bash
npx -y @modelcontextprotocol/server-github
```

### 配置 MCP Host

以 Claude Desktop 为例，编辑配置文件：

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/projects"],
      "env": {}
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
      "env": {
        "PGPASSWORD": "your-password"
      }
    }
  }
}
```

## 开发 MCP Server

### 使用 TypeScript SDK

```bash
npm install @modelcontextprotocol/sdk
```

### 基础 Server 示例

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// 创建 Server 实例
const server = new Server(
  { name: "my-mcp-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// 定义可用工具
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "get_weather",
      description: "获取指定城市的天气信息",
      inputSchema: {
        type: "object",
        properties: {
          city: { type: "string", description: "城市名称" }
        },
        required: ["city"]
      }
    }
  ]
}));

// 处理工具调用
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "get_weather") {
    const { city } = request.params.arguments as { city: string };
    // 实际应用中调用天气 API
    const weather = await fetchWeather(city);
    return {
      content: [{ type: "text", text: JSON.stringify(weather) }]
    };
  }
  throw new Error(`Unknown tool: ${request.params.name}`);
});

// 启动 Server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
```


### 使用 Python SDK

```bash
pip install mcp
```

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 创建 Server
server = Server("my-mcp-server")

# 定义工具
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_docs",
            description="搜索文档内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        )
    ]

# 处理工具调用
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_docs":
        results = search_documents(arguments["query"])
        return [TextContent(type="text", text=str(results))]
    raise ValueError(f"Unknown tool: {name}")

# 启动 Server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 添加 Resources 支持

```typescript
import { ListResourcesRequestSchema, ReadResourceRequestSchema } from "@modelcontextprotocol/sdk/types.js";

// 列出可用资源
server.setRequestHandler(ListResourcesRequestSchema, async () => ({
  resources: [
    {
      uri: "config://app/settings",
      name: "应用配置",
      mimeType: "application/json"
    }
  ]
}));

// 读取资源内容
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  if (request.params.uri === "config://app/settings") {
    const config = await loadConfig();
    return {
      contents: [{
        uri: request.params.uri,
        mimeType: "application/json",
        text: JSON.stringify(config, null, 2)
      }]
    };
  }
  throw new Error(`Unknown resource: ${request.params.uri}`);
});
```

## 常用 MCP Server

| Server                | 功能           | 安装命令                                        |
| --------------------- | -------------- | ----------------------------------------------- |
| **filesystem**        | 文件系统访问   | `npx @modelcontextprotocol/server-filesystem`   |
| **postgres**          | PostgreSQL     | `npx @modelcontextprotocol/server-postgres`     |
| **sqlite**            | SQLite 数据库  | `npx @modelcontextprotocol/server-sqlite`       |
| **github**            | GitHub API     | `npx @modelcontextprotocol/server-github`       |
| **slack**             | Slack 集成     | `npx @modelcontextprotocol/server-slack`        |
| **puppeteer**         | 浏览器自动化   | `npx @modelcontextprotocol/server-puppeteer`    |
| **brave-search**      | Brave 搜索     | `npx @modelcontextprotocol/server-brave-search` |
| **aws-documentation** | AWS 文档搜索   | `uvx awslabs.aws-documentation-mcp-server`      |

## MCP vs Function Calling

| 特性         | Function Calling | MCP          |
| ------------ | ---------------- | ------------ |
| **定位**     | 模型原生能力     | 连接协议标准 |
| **工具定义** | API 私有格式     | 统一标准格式 |
| **可移植性** | 绑定特定模型     | 跨平台通用   |
| **资源访问** | 不支持           | 原生支持     |
| **提示词模板** | 不支持         | 原生支持     |
| **复杂度**   | 较简单           | 功能更丰富   |

:::tip 选择建议
- 简单场景、单一模型：使用 Function Calling
- 复杂场景、多平台支持：使用 MCP
- 两者可以结合使用：MCP Server 内部可以使用 Function Calling
:::

## 安全最佳实践

1. **最小权限原则**：只暴露必要的工具和资源
2. **输入验证**：严格验证所有工具参数
3. **访问控制**：实现基于用户/角色的权限控制
4. **审计日志**：记录所有工具调用和资源访问
5. **敏感数据保护**：避免在响应中暴露敏感信息

```typescript
// 输入验证示例
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  // 验证工具名称
  if (!ALLOWED_TOOLS.includes(name)) {
    throw new Error(`Tool not allowed: ${name}`);
  }
  
  // 验证参数
  const validated = validateArgs(name, args);
  if (!validated.success) {
    throw new Error(`Invalid arguments: ${validated.error}`);
  }
  
  // 记录审计日志
  await logToolCall(name, args, request.context?.user);
  
  return executeToolSafely(name, validated.data);
});
```

## 调试技巧

### 使用 MCP Inspector

```bash
npx @modelcontextprotocol/inspector npx @modelcontextprotocol/server-filesystem /tmp
```

### 启用详细日志

```typescript
const server = new Server(
  { name: "my-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// 添加请求日志
server.onRequest = (request) => {
  console.error(`[MCP] Request: ${JSON.stringify(request)}`);
};
```

## 延伸阅读

- [MCP 官方文档](https://modelcontextprotocol.io/introduction)
- [MCP GitHub 仓库](https://github.com/modelcontextprotocol)
- [MCP Server 示例集合](https://github.com/modelcontextprotocol/servers)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
