---
sidebar_position: 3
title: HTTP 与网络
---

# HTTP 与网络

> [!TIP]
> 理解 HTTP 协议和网络请求是前端开发的必备技能。

## 🌐 HTTP 基础

### 请求结构

```
GET /api/users HTTP/1.1
Host: example.com
Content-Type: application/json
Authorization: Bearer token123

{请求体}
```

### 响应结构

```
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: max-age=3600

{响应体}
```

### HTTP 方法

| 方法   | 用途                 | 特点         |
| ------ | -------------------- | ------------ |
| GET    | 获取资源             | 幂等，可缓存 |
| POST   | 创建资源             | 非幂等       |
| PUT    | 更新资源（完整替换） | 幂等         |
| PATCH  | 部分更新             | 非幂等       |
| DELETE | 删除资源             | 幂等         |

### 状态码

| 范围 | 含义       | 常见               |
| ---- | ---------- | ------------------ |
| 2xx  | 成功       | 200, 201, 204      |
| 3xx  | 重定向     | 301, 302, 304      |
| 4xx  | 客户端错误 | 400, 401, 403, 404 |
| 5xx  | 服务端错误 | 500, 502, 503      |

## 📡 Fetch API

### 基本请求

```javascript
// GET 请求
const response = await fetch("/api/users");
const data = await response.json();

// POST 请求
const response = await fetch("/api/users", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ name: "Alice" }),
});
```

### 完整配置

```javascript
const response = await fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: "Bearer token",
  },
  body: JSON.stringify(data),
  mode: "cors", // cors, no-cors, same-origin
  credentials: "include", // include, same-origin, omit
  cache: "no-cache", // default, no-cache, reload
  signal: controller.signal, // 取消请求
});
```

### 响应处理

```javascript
const response = await fetch(url);

// 检查状态
if (!response.ok) {
  throw new Error(`HTTP ${response.status}`);
}

// 不同格式
const json = await response.json();
const text = await response.text();
const blob = await response.blob();
const buffer = await response.arrayBuffer();
const form = await response.formData();
```

### 封装请求

```javascript
async function request(url, options = {}) {
  const defaultOptions = {
    headers: {
      "Content-Type": "application/json",
    },
  };

  const response = await fetch(url, {
    ...defaultOptions,
    ...options,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  return response.json();
}

// 使用
const users = await request("/api/users");
const newUser = await request("/api/users", {
  method: "POST",
  body: JSON.stringify({ name: "Alice" }),
});
```

### 取消请求

```javascript
const controller = new AbortController();

fetch(url, { signal: controller.signal })
  .then((response) => response.json())
  .catch((error) => {
    if (error.name === "AbortError") {
      console.log("请求被取消");
    }
  });

// 取消
controller.abort();
```

## 🔄 XMLHttpRequest

传统的请求方式，某些场景仍在使用：

```javascript
const xhr = new XMLHttpRequest();
xhr.open("GET", "/api/users");
xhr.setRequestHeader("Content-Type", "application/json");

xhr.onload = function () {
  if (xhr.status === 200) {
    const data = JSON.parse(xhr.responseText);
    console.log(data);
  }
};

xhr.onerror = function () {
  console.error("请求失败");
};

xhr.send();
```

### 上传进度

```javascript
xhr.upload.onprogress = function (event) {
  if (event.lengthComputable) {
    const percent = (event.loaded / event.total) * 100;
    console.log(`上传进度: ${percent}%`);
  }
};
```

## 🌍 跨域 CORS

### 同源策略

同源 = 协议 + 域名 + 端口都相同

```
http://example.com/page  ─┬─ 同源
http://example.com/other ─┘

http://example.com  ─┬─ 不同源（端口不同）
http://example.com:8080 ─┘
```

### CORS 机制

```
浏览器                           服务器
  │                               │
  │─── 简单请求 ──────────────────>│
  │<── 响应 + CORS 头 ────────────│
  │                               │
  │─── OPTIONS 预检 ─────────────>│  ← 复杂请求前
  │<── 允许的方法/头 ─────────────│
  │─── 实际请求 ─────────────────>│
  │<── 响应 ─────────────────────│
```

### 简单请求条件

- 方法：GET, HEAD, POST
- 头部：只有简单头部（Content-Type 仅限 text/plain, multipart/form-data, application/x-www-form-urlencoded）

### 服务端响应头

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Allow-Credentials: true
Access-Control-Max-Age: 86400
```

## 📤 文件上传

### FormData

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);
formData.append("name", "document.pdf");

await fetch("/api/upload", {
  method: "POST",
  body: formData, // 不设置 Content-Type，浏览器自动处理
});
```

### 多文件上传

```javascript
const formData = new FormData();
for (const file of fileInput.files) {
  formData.append("files", file);
}

await fetch("/api/upload", {
  method: "POST",
  body: formData,
});
```

## ⚡ 请求优化

### 请求缓存

```javascript
// 使用 Cache API
const cache = await caches.open("api-cache");

// 缓存策略
async function cachedFetch(url) {
  const cached = await cache.match(url);
  if (cached) return cached;

  const response = await fetch(url);
  cache.put(url, response.clone());
  return response;
}
```

### 请求重试

```javascript
async function fetchWithRetry(url, options = {}, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      return await fetch(url, options);
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise((r) => setTimeout(r, 1000 * (i + 1)));
    }
  }
}
```

### 请求去重

```javascript
const pending = new Map();

async function dedupeFetch(url) {
  if (pending.has(url)) {
    return pending.get(url);
  }

  const promise = fetch(url).finally(() => {
    pending.delete(url);
  });

  pending.set(url, promise);
  return promise;
}
```

## 💡 最佳实践

### 1. 统一错误处理

```javascript
async function request(url, options) {
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message);
    }
    return response.json();
  } catch (error) {
    console.error("请求失败:", error);
    throw error;
  }
}
```

### 2. 请求超时

```javascript
function fetchWithTimeout(url, timeout = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  return fetch(url, { signal: controller.signal }).finally(() =>
    clearTimeout(timeoutId)
  );
}
```

## 🔗 相关资源

- [异步编程](/docs/frontend/javascript/async)
- [浏览器存储](/docs/frontend/browser/storage)

---

**下一步**：学习 [前端性能优化](/docs/frontend/advanced/performance) 提升应用速度。
