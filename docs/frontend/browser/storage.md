---
sidebar_position: 2
title: 浏览器存储
---

# 浏览器存储

> [!TIP]
> 浏览器提供多种存储方式，用于在客户端保存数据。

## 🍪 Cookie

Cookie 主要用于身份验证和服务端会话。

### 基本操作

```javascript
// 设置 Cookie
document.cookie = "username=Alice";
document.cookie = "theme=dark; max-age=86400"; // 1天

// 读取 Cookie
console.log(document.cookie); // 'username=Alice; theme=dark'

// 解析 Cookie
function getCookie(name) {
  const cookies = document.cookie.split("; ");
  for (const cookie of cookies) {
    const [key, value] = cookie.split("=");
    if (key === name) return decodeURIComponent(value);
  }
  return null;
}

// 删除 Cookie（设置过期时间为过去）
document.cookie = "username=; expires=Thu, 01 Jan 1970 00:00:00 GMT";
```

### Cookie 属性

```javascript
document.cookie = `
  token=abc123;
  path=/;
  domain=.example.com;
  max-age=604800;
  secure;
  samesite=strict
`;
```

| 属性       | 说明                       |
| ---------- | -------------------------- |
| `path`     | Cookie 可用路径            |
| `domain`   | Cookie 可用域名            |
| `max-age`  | 有效期（秒）               |
| `expires`  | 过期日期                   |
| `secure`   | 仅 HTTPS 传输              |
| `httpOnly` | 禁止 JS 访问（服务端设置） |
| `samesite` | 跨站限制 (strict/lax/none) |

## 📦 LocalStorage

持久化存储，除非主动删除，数据永久保存。

### 基本操作

```javascript
// 存储
localStorage.setItem("user", JSON.stringify({ name: "Alice" }));
localStorage.theme = "dark"; // 简写

// 读取
const user = JSON.parse(localStorage.getItem("user"));
const theme = localStorage.theme;

// 删除
localStorage.removeItem("user");

// 清空
localStorage.clear();

// 遍历
for (let i = 0; i < localStorage.length; i++) {
  const key = localStorage.key(i);
  console.log(key, localStorage.getItem(key));
}
```

### 存储对象

```javascript
// 封装工具函数
const storage = {
  get(key, defaultValue = null) {
    try {
      const value = localStorage.getItem(key);
      return value ? JSON.parse(value) : defaultValue;
    } catch {
      return defaultValue;
    }
  },

  set(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
  },

  remove(key) {
    localStorage.removeItem(key);
  },
};

// 使用
storage.set("settings", { theme: "dark", lang: "zh" });
const settings = storage.get("settings", {});
```

## 📋 SessionStorage

会话存储，页面关闭后数据清除。

```javascript
// API 与 LocalStorage 完全相同
sessionStorage.setItem("tempData", "value");
const data = sessionStorage.getItem("tempData");
sessionStorage.removeItem("tempData");
```

### 使用场景

- 表单临时数据
- 页面间传递数据
- 一次性操作状态

## 📊 对比

| 特性     | Cookie     | LocalStorage | SessionStorage |
| -------- | ---------- | ------------ | -------------- |
| 存储大小 | ~4KB       | ~5MB         | ~5MB           |
| 过期时间 | 可设置     | 永久         | 页面关闭       |
| 自动发送 | 每次请求   | 否           | 否             |
| 作用域   | 路径+域名  | 同源         | 同源+同标签    |
| API      | 字符串操作 | 简洁 API     | 简洁 API       |

## 💾 IndexedDB

浏览器内置的 NoSQL 数据库，适合存储大量结构化数据。

### 基本使用

```javascript
// 打开数据库
const request = indexedDB.open("MyDB", 1);

request.onerror = () => console.error("打开失败");

request.onupgradeneeded = (event) => {
  const db = event.target.result;

  // 创建对象仓库（类似表）
  if (!db.objectStoreNames.contains("users")) {
    const store = db.createObjectStore("users", { keyPath: "id" });
    store.createIndex("name", "name", { unique: false });
  }
};

request.onsuccess = (event) => {
  const db = event.target.result;

  // 添加数据
  const tx = db.transaction("users", "readwrite");
  const store = tx.objectStore("users");
  store.add({ id: 1, name: "Alice", age: 25 });

  // 读取数据
  const getRequest = store.get(1);
  getRequest.onsuccess = () => {
    console.log(getRequest.result);
  };
};
```

### 封装 Promise

```javascript
function openDB(name, version, upgradeCallback) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, version);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (e) => upgradeCallback(e.target.result);
  });
}

// 使用
const db = await openDB("MyDB", 1, (db) => {
  db.createObjectStore("users", { keyPath: "id" });
});
```

## 🔐 存储安全

### 敏感数据处理

```javascript
// ❌ 不要存储敏感信息
localStorage.setItem("password", "123456");

// ✅ 敏感数据应该
// 1. 使用 httpOnly Cookie（服务端设置）
// 2. 使用 sessionStorage 存临时 token
// 3. 必要时加密存储
```

### 存储监听

```javascript
// 监听其他标签页的存储变化
window.addEventListener("storage", (event) => {
  console.log("Key:", event.key);
  console.log("Old:", event.oldValue);
  console.log("New:", event.newValue);
  console.log("URL:", event.url);
});
```

## 💡 最佳实践

### 1. 选择合适的存储方式

```javascript
// 身份验证 → Cookie (httpOnly)
// 用户偏好 → LocalStorage
// 临时表单 → SessionStorage
// 大量数据 → IndexedDB
```

### 2. 处理存储异常

```javascript
try {
  localStorage.setItem("key", "value");
} catch (e) {
  if (e.name === "QuotaExceededError") {
    console.error("存储已满");
    // 清理旧数据
  }
}
```

### 3. 数据版本管理

```javascript
const STORAGE_VERSION = "1.0";

function migrateStorage() {
  const version = localStorage.getItem("storageVersion");
  if (version !== STORAGE_VERSION) {
    // 数据迁移逻辑
    localStorage.clear();
    localStorage.setItem("storageVersion", STORAGE_VERSION);
  }
}
```

## 🔗 相关资源

- [浏览器原理](/docs/frontend/browser/)
- [HTTP 网络](/docs/frontend/browser/network)

---

**下一步**：学习 [HTTP 网络](/docs/frontend/browser/network) 了解数据传输。
