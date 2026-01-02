---
sidebar_position: 4
title: Web Workers
---

# Web Workers

> [!TIP]
> Web Workers 让你在后台线程运行脚本，不阻塞主线程，保持页面流畅响应。

## 🎯 为什么需要 Workers？

JavaScript 是单线程的，复杂计算会阻塞 UI：

```javascript
// ❌ 阻塞主线程
function heavyTask() {
  let result = 0;
  for (let i = 0; i < 1e9; i++) {
    result += Math.sqrt(i);
  }
  return result;
}

button.onclick = () => {
  heavyTask(); // 页面卡死几秒
};
```

使用 Worker 解决：

```javascript
// ✅ 后台线程处理
const worker = new Worker("worker.js");

button.onclick = () => {
  worker.postMessage("start");
};

worker.onmessage = (e) => {
  console.log("结果:", e.data); // UI 保持流畅
};
```

## 📦 Dedicated Worker

最常用的 Worker 类型，专属于创建它的脚本。

### 创建 Worker

```javascript
// main.js
const worker = new Worker("worker.js");

// 发送消息
worker.postMessage({ type: "calculate", data: [1, 2, 3, 4, 5] });

// 接收消息
worker.onmessage = (event) => {
  console.log("收到结果:", event.data);
};

// 错误处理
worker.onerror = (error) => {
  console.error("Worker 错误:", error.message);
};

// 终止 Worker
worker.terminate();
```

```javascript
// worker.js
self.onmessage = (event) => {
  const { type, data } = event.data;

  if (type === "calculate") {
    const result = data.reduce((sum, n) => sum + n, 0);
    self.postMessage(result);
  }
};
```

### 传输大数据

```javascript
// 复制数据（较慢）
worker.postMessage({ largeArray: array });

// 转移所有权（快速，原数组不可用）
worker.postMessage(buffer, [buffer]);

// 检查是否可转移
const data = new Float32Array(1000000);
worker.postMessage(data.buffer, [data.buffer]);
// data.buffer 现在为空
```

## 🔄 Shared Worker

多个页面/脚本共享同一个 Worker。

```javascript
// main.js (多个页面可共享)
const shared = new SharedWorker("shared-worker.js");

shared.port.onmessage = (e) => {
  console.log("收到:", e.data);
};

shared.port.postMessage("hello");
shared.port.start();
```

```javascript
// shared-worker.js
const connections = [];

self.onconnect = (e) => {
  const port = e.ports[0];
  connections.push(port);

  port.onmessage = (event) => {
    // 广播给所有连接
    connections.forEach((p) => {
      p.postMessage(`广播: ${event.data}`);
    });
  };

  port.start();
};
```

## ⚡ Service Worker

拦截网络请求，实现离线缓存。

### 注册

```javascript
// main.js
if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register("/sw.js")
    .then((reg) => console.log("SW 注册成功"))
    .catch((err) => console.log("SW 注册失败:", err));
}
```

### 缓存策略

```javascript
// sw.js
const CACHE_NAME = "v1";
const ASSETS = ["/", "/styles.css", "/app.js", "/offline.html"];

// 安装时缓存
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

// 请求时优先缓存
self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((cached) => {
      return cached || fetch(event.request);
    })
  );
});

// 激活时清理旧缓存
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
  );
});
```

### 缓存策略对比

| 策略                   | 说明                   |
| ---------------------- | ---------------------- |
| Cache First            | 优先缓存，适合静态资源 |
| Network First          | 优先网络，适合动态内容 |
| Stale While Revalidate | 先返回缓存，后台更新   |
| Network Only           | 只用网络               |
| Cache Only             | 只用缓存               |

## 🎮 实用示例

### 图片处理

```javascript
// main.js
const imageWorker = new Worker("image-worker.js");

imageWorker.postMessage({ imageData, filter: "grayscale" });

imageWorker.onmessage = (e) => {
  ctx.putImageData(e.data, 0, 0);
};
```

```javascript
// image-worker.js
self.onmessage = (e) => {
  const { imageData, filter } = e.data;
  const data = imageData.data;

  if (filter === "grayscale") {
    for (let i = 0; i < data.length; i += 4) {
      const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i] = data[i + 1] = data[i + 2] = gray;
    }
  }

  self.postMessage(imageData);
};
```

### 大数据排序

```javascript
// main.js
const sortWorker = new Worker("sort-worker.js");

// 发送大数组
const largeArray = new Array(1000000).fill(0).map(() => Math.random());
sortWorker.postMessage(largeArray);

sortWorker.onmessage = (e) => {
  console.log("排序完成", e.data);
};
```

```javascript
// sort-worker.js
self.onmessage = (e) => {
  const sorted = e.data.sort((a, b) => a - b);
  self.postMessage(sorted);
};
```

## ⚠️ Worker 限制

Workers 无法访问：

- DOM（`document`、`window`）
- 父页面的变量
- 某些 API（`alert`、`confirm`）

Workers 可以使用：

- `fetch`、`XMLHttpRequest`
- `setTimeout`、`setInterval`
- `IndexedDB`、`Cache API`
- `importScripts()` 加载脚本

```javascript
// 在 Worker 中加载库
importScripts("lodash.min.js", "utils.js");
```

## 💡 最佳实践

1. **重计算任务** → Dedicated Worker
2. **跨页面通信** → Shared Worker
3. **离线缓存** → Service Worker
4. **大数据传输** → 使用 Transferable Objects
5. **及时销毁** → 任务完成后调用 `terminate()`

## 🔗 相关资源

- [浏览器原理](/docs/frontend/browser/)
- [性能优化](/docs/frontend/advanced/performance)

---

**恭喜！** 你已完成前端进阶学习。继续探索 [React](/docs/react) 开发现代应用！
