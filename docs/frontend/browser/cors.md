---
sidebar_position: 5
title: 跨域详解
---

# 跨域详解

> [!TIP]
> 跨域是前端开发中的常见问题，理解其原理和解决方案是必备技能。

## 🎯 什么是跨域？

浏览器的**同源策略**限制了不同源之间的资源访问。

### 同源的定义

| 比较项                                | 是否同源 | 原因       |
| ------------------------------------- | -------- | ---------- |
| `http://a.com` vs `http://a.com`      | ✅       | 完全相同   |
| `http://a.com` vs `https://a.com`     | ❌       | 协议不同   |
| `http://a.com` vs `http://b.com`      | ❌       | 域名不同   |
| `http://a.com` vs `http://a.com:8080` | ❌       | 端口不同   |
| `http://a.com` vs `http://www.a.com`  | ❌       | 子域名不同 |

## 📦 解决方案

### 1. CORS（推荐）

服务端设置响应头允许跨域。

```javascript
// Node.js Express
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "http://example.com");
  res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.header("Access-Control-Allow-Credentials", "true");

  // 预检请求
  if (req.method === "OPTIONS") {
    return res.sendStatus(200);
  }
  next();
});
```

#### 简单请求 vs 预检请求

**简单请求**（直接发送）：

- 方法：GET、HEAD、POST
- Content-Type：text/plain、multipart/form-data、application/x-www-form-urlencoded
- 无自定义头

**预检请求**（先发 OPTIONS）：

```
OPTIONS /api/data HTTP/1.1
Origin: http://example.com
Access-Control-Request-Method: PUT
Access-Control-Request-Headers: X-Custom-Header
```

```
HTTP/1.1 200 OK
Access-Control-Allow-Origin: http://example.com
Access-Control-Allow-Methods: GET, PUT, POST
Access-Control-Allow-Headers: X-Custom-Header
Access-Control-Max-Age: 86400
```

#### 携带凭证

```javascript
// 前端
fetch("http://api.example.com/data", {
  credentials: "include", // 携带 Cookie
});

// 服务端
Access-Control-Allow-Credentials: true
Access-Control-Allow-Origin: http://example.com  // 不能是 *
```

### 2. 代理服务器

开发环境最常用。

```javascript
// vite.config.js
export default {
  server: {
    proxy: {
      "/api": {
        target: "http://api.example.com",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
};
```

```javascript
// webpack.config.js
module.exports = {
  devServer: {
    proxy: {
      "/api": {
        target: "http://api.example.com",
        changeOrigin: true,
        pathRewrite: { "^/api": "" },
      },
    },
  },
};
```

### 3. JSONP

利用 `<script>` 标签不受同源策略限制。

```javascript
function jsonp(url, callback) {
  return new Promise((resolve) => {
    const callbackName = `jsonp_${Date.now()}`;

    window[callbackName] = (data) => {
      delete window[callbackName];
      document.body.removeChild(script);
      resolve(data);
    };

    const script = document.createElement("script");
    script.src = `${url}?callback=${callbackName}`;
    document.body.appendChild(script);
  });
}

// 使用
const data = await jsonp("http://api.example.com/data");
```

> [!WARNING]
> JSONP 只支持 GET 请求，且存在安全风险，现代项目不推荐使用。

### 4. postMessage

跨窗口通信。

```javascript
// 父页面
const iframe = document.querySelector("iframe");

iframe.onload = () => {
  iframe.contentWindow.postMessage({ type: "getData" }, "http://other.com");
};

window.addEventListener("message", (event) => {
  if (event.origin !== "http://other.com") return;
  console.log("收到数据:", event.data);
});
```

```javascript
// iframe 页面
window.addEventListener("message", (event) => {
  if (event.origin !== "http://parent.com") return;

  if (event.data.type === "getData") {
    event.source.postMessage({ result: "data" }, event.origin);
  }
});
```

### 5. WebSocket

WebSocket 不受同源策略限制。

```javascript
const ws = new WebSocket("wss://api.example.com/socket");

ws.onopen = () => {
  ws.send(JSON.stringify({ type: "subscribe" }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("收到:", data);
};
```

### 6. Nginx 反向代理

生产环境常用。

```nginx
server {
    listen 80;
    server_name example.com;

    location / {
        root /var/www/html;
    }

    location /api/ {
        proxy_pass http://api.backend.com/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 方案对比

| 方案        | 场景         | 优点         | 缺点               |
| ----------- | ------------ | ------------ | ------------------ |
| CORS        | 标准方案     | 安全、标准   | 需要服务端配合     |
| 代理        | 开发环境     | 简单、无侵入 | 只适合开发环境     |
| JSONP       | 兼容老浏览器 | 兼容性好     | 只支持 GET、不安全 |
| postMessage | 跨窗口通信   | 灵活         | 需要两端配合       |
| WebSocket   | 实时通信     | 全双工       | 协议不同           |
| Nginx       | 生产环境     | 高效         | 需要运维配置       |

## 💡 常见问题

### Cookie 跨域

```javascript
// 前端
fetch(url, { credentials: "include" });

// 服务端
Set-Cookie: token=xxx; SameSite=None; Secure
```

### localStorage 跨域

通过 postMessage + iframe 实现：

```javascript
// 主页面
function getStorageFromOther(domain, key) {
  return new Promise((resolve) => {
    const iframe = document.createElement("iframe");
    iframe.src = `${domain}/storage.html`;
    iframe.style.display = "none";

    iframe.onload = () => {
      iframe.contentWindow.postMessage({ type: "get", key }, domain);
    };

    window.addEventListener("message", function handler(e) {
      if (e.origin === domain) {
        resolve(e.data);
        window.removeEventListener("message", handler);
        document.body.removeChild(iframe);
      }
    });

    document.body.appendChild(iframe);
  });
}
```

## 🔗 相关资源

- [HTTP 网络](/docs/frontend/browser/network)
- [前端安全](/docs/frontend/advanced/security)

---

**下一步**：学习 [调试技巧](/docs/frontend/browser/debugging) 提升开发效率。
