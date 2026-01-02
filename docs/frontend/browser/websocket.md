---
sidebar_position: 7
title: WebSocket
---

# WebSocket 实时通信

> [!TIP]
> WebSocket 提供全双工通信通道，适合实时应用如聊天、游戏、实时数据推送。

## 🎯 什么是 WebSocket？

WebSocket 是一种在单个 TCP 连接上进行全双工通信的协议。

### HTTP vs WebSocket

| 特性     | HTTP           | WebSocket    |
| -------- | -------------- | ------------ |
| 连接     | 短连接         | 长连接       |
| 通信方式 | 请求-响应      | 双向推送     |
| 开销     | 每次请求带头部 | 建立后开销小 |
| 适用场景 | 普通请求       | 实时通信     |

## 📦 基础用法

### 创建连接

```javascript
// 创建 WebSocket 连接
const ws = new WebSocket("wss://example.com/socket");

// 连接成功
ws.onopen = () => {
  console.log("连接成功");
  ws.send("Hello Server!");
};

// 收到消息
ws.onmessage = (event) => {
  console.log("收到消息:", event.data);
};

// 连接关闭
ws.onclose = (event) => {
  console.log("连接关闭:", event.code, event.reason);
};

// 发生错误
ws.onerror = (error) => {
  console.error("错误:", error);
};
```

### 发送消息

```javascript
// 发送文本
ws.send("Hello");

// 发送 JSON
ws.send(JSON.stringify({ type: "message", data: "Hello" }));

// 发送二进制
const buffer = new ArrayBuffer(8);
ws.send(buffer);

// 检查连接状态
if (ws.readyState === WebSocket.OPEN) {
  ws.send("消息");
}
```

### 连接状态

```javascript
ws.readyState === 0; // CONNECTING 连接中
ws.readyState === 1; // OPEN 已连接
ws.readyState === 2; // CLOSING 关闭中
ws.readyState === 3; // CLOSED 已关闭
```

## 🔧 封装 WebSocket

```javascript
class WebSocketClient {
  constructor(url, options = {}) {
    this.url = url;
    this.options = {
      reconnectInterval: 3000,
      maxReconnectAttempts: 5,
      ...options,
    };

    this.ws = null;
    this.reconnectAttempts = 0;
    this.handlers = {};

    this.connect();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log("WebSocket 已连接");
      this.reconnectAttempts = 0;
      this.emit("open");
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.emit(data.type, data.payload);
      } catch {
        this.emit("message", event.data);
      }
    };

    this.ws.onclose = (event) => {
      console.log("WebSocket 已关闭");
      this.emit("close", event);
      this.reconnect();
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket 错误:", error);
      this.emit("error", error);
    };
  }

  reconnect() {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.log("达到最大重连次数");
      return;
    }

    this.reconnectAttempts++;
    console.log(`${this.options.reconnectInterval}ms 后重连...`);

    setTimeout(() => {
      this.connect();
    }, this.options.reconnectInterval);
  }

  send(type, payload) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, payload }));
    }
  }

  on(event, handler) {
    if (!this.handlers[event]) {
      this.handlers[event] = [];
    }
    this.handlers[event].push(handler);
  }

  emit(event, data) {
    if (this.handlers[event]) {
      this.handlers[event].forEach((handler) => handler(data));
    }
  }

  close() {
    this.ws.close();
  }
}

// 使用
const client = new WebSocketClient("wss://example.com/socket");

client.on("open", () => {
  client.send("subscribe", { channel: "news" });
});

client.on("news", (data) => {
  console.log("新闻:", data);
});
```

## 💓 心跳机制

保持连接活跃，检测断线。

```javascript
class WebSocketWithHeartbeat {
  constructor(url) {
    this.url = url;
    this.heartbeatInterval = 30000;
    this.heartbeatTimer = null;
    this.connect();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.startHeartbeat();
    };

    this.ws.onmessage = (event) => {
      if (event.data === "pong") {
        // 收到心跳响应
        return;
      }
      // 处理其他消息
    };

    this.ws.onclose = () => {
      this.stopHeartbeat();
    };
  }

  startHeartbeat() {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.send("ping");
      }
    }, this.heartbeatInterval);
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }
}
```

## 🎮 实际应用

### 聊天室

```javascript
const chat = new WebSocketClient("wss://chat.example.com");

// 发送消息
function sendMessage(text) {
  chat.send("chat", {
    text,
    user: currentUser,
    timestamp: Date.now(),
  });
}

// 接收消息
chat.on("chat", (message) => {
  appendMessage(message);
});

// 用户列表更新
chat.on("users", (users) => {
  updateUserList(users);
});
```

### 实时数据

```javascript
const stocks = new WebSocketClient("wss://stocks.example.com");

stocks.on("open", () => {
  stocks.send("subscribe", { symbols: ["AAPL", "GOOGL"] });
});

stocks.on("price", ({ symbol, price }) => {
  updateStockPrice(symbol, price);
});
```

## 🔐 安全考虑

```javascript
// 使用 wss:// 加密连接
const ws = new WebSocket("wss://example.com/socket");

// 携带认证信息
const ws = new WebSocket("wss://example.com/socket?token=" + authToken);

// 或在连接后发送认证
ws.onopen = () => {
  ws.send(
    JSON.stringify({
      type: "auth",
      token: authToken,
    })
  );
};
```

## 📊 轮询 vs SSE vs WebSocket

| 方案      | 方向            | 实时性 | 复杂度 | 适用场景 |
| --------- | --------------- | ------ | ------ | -------- |
| 轮询      | 客户端 → 服务端 | 低     | 低     | 简单通知 |
| SSE       | 服务端 → 客户端 | 中     | 中     | 单向推送 |
| WebSocket | 双向            | 高     | 高     | 实时交互 |

## 💡 最佳实践

1. **使用 wss://** - 生产环境必须加密
2. **实现重连** - 网络不稳定时自动恢复
3. **心跳检测** - 检测连接状态
4. **消息确认** - 重要消息需要确认机制
5. **优雅降级** - 不支持时回退到轮询

## 🔗 相关资源

- [HTTP 网络](/docs/frontend/browser/network)
- [跨域详解](/docs/frontend/browser/cors)

---

**下一步**：学习 [前端监控](/docs/frontend/advanced/monitoring) 保障应用质量。
