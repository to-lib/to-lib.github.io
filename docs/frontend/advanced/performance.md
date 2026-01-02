---
sidebar_position: 1
title: 性能优化
---

# 前端性能优化

> [!TIP]
> 性能优化直接影响用户体验和业务指标。页面加载每慢 1 秒，转化率可能下降 7%。

## 📊 核心性能指标 (Core Web Vitals)

| 指标    | 含义             | 目标    |
| ------- | ---------------- | ------- |
| **LCP** | 最大内容绘制时间 | < 2.5s  |
| **FID** | 首次输入延迟     | < 100ms |
| **CLS** | 累积布局偏移     | < 0.1   |

### 测量工具

- Chrome DevTools Performance
- Lighthouse
- WebPageTest
- web.dev/measure

## ⚡ 加载优化

### 资源压缩

```bash
# 图片压缩
npx imagemin src/images/* --out-dir=dist/images

# JS/CSS 压缩（构建工具自动处理）
```

### 代码分割

```javascript
// React 动态导入
const HeavyComponent = React.lazy(() => import("./HeavyComponent"));

// 路由级分割
const routes = [
  {
    path: "/dashboard",
    component: () => import("./pages/Dashboard"),
  },
];
```

### 资源预加载

```html
<!-- 预加载关键资源 -->
<link rel="preload" href="font.woff2" as="font" crossorigin />
<link rel="preload" href="hero.jpg" as="image" />

<!-- 预连接第三方域名 -->
<link rel="preconnect" href="https://api.example.com" />
<link rel="dns-prefetch" href="https://cdn.example.com" />

<!-- 预获取下一页 -->
<link rel="prefetch" href="/next-page.js" />
```

### 延迟加载

```html
<!-- 图片懒加载 -->
<img
  src="placeholder.jpg"
  data-src="real-image.jpg"
  loading="lazy"
  alt="描述"
/>

<!-- 使用 Intersection Observer -->
```

```javascript
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      const img = entry.target;
      img.src = img.dataset.src;
      observer.unobserve(img);
    }
  });
});

document.querySelectorAll("img[data-src]").forEach((img) => {
  observer.observe(img);
});
```

## 🎨 渲染优化

### 避免强制同步布局

```javascript
// ❌ 强制同步布局
elements.forEach((el) => {
  el.style.width = container.offsetWidth + "px";
});

// ✅ 先读后写
const width = container.offsetWidth;
elements.forEach((el) => {
  el.style.width = width + "px";
});
```

### 使用 transform 和 opacity

```css
/* ❌ 触发重排 */
.animate {
  left: 100px;
  top: 100px;
}

/* ✅ 只触发合成 */
.animate {
  transform: translate(100px, 100px);
}
```

### 虚拟列表

```javascript
// 只渲染可见区域的列表项
function VirtualList({ items, itemHeight, containerHeight }) {
  const [scrollTop, setScrollTop] = useState(0);

  const startIndex = Math.floor(scrollTop / itemHeight);
  const endIndex = Math.min(
    startIndex + Math.ceil(containerHeight / itemHeight) + 1,
    items.length
  );

  const visibleItems = items.slice(startIndex, endIndex);

  return (
    <div
      style={{ height: containerHeight, overflow: "auto" }}
      onScroll={(e) => setScrollTop(e.target.scrollTop)}
    >
      <div style={{ height: items.length * itemHeight }}>
        <div style={{ transform: `translateY(${startIndex * itemHeight}px)` }}>
          {visibleItems.map((item) => (
            <div key={item.id} style={{ height: itemHeight }}>
              {item.content}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

## 📦 缓存策略

### HTTP 缓存

```
# 静态资源 - 长期缓存 + hash
Cache-Control: max-age=31536000

# HTML - 需要验证
Cache-Control: no-cache

# API - 短期缓存
Cache-Control: max-age=60
```

### Service Worker

```javascript
// 安装时缓存静态资源
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open("v1").then((cache) => {
      return cache.addAll(["/", "/styles.css", "/app.js"]);
    })
  );
});

// 请求时优先使用缓存
self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
```

## 🖼️ 图片优化

### 现代格式

```html
<picture>
  <source srcset="image.avif" type="image/avif" />
  <source srcset="image.webp" type="image/webp" />
  <img src="image.jpg" alt="描述" />
</picture>
```

### 响应式图片

```html
<img
  srcset="small.jpg 300w, medium.jpg 600w, large.jpg 1200w"
  sizes="(max-width: 600px) 300px,
         (max-width: 1200px) 600px,
         1200px"
  src="medium.jpg"
  alt="描述"
/>
```

## ⚙️ JavaScript 优化

### 防抖和节流

```javascript
// 防抖 - 停止触发后执行
function debounce(fn, delay) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

// 节流 - 固定频率执行
function throttle(fn, limit) {
  let inThrottle;
  return function (...args) {
    if (!inThrottle) {
      fn.apply(this, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

// 使用
window.addEventListener("scroll", throttle(handleScroll, 100));
input.addEventListener("input", debounce(search, 300));
```

### Web Worker

```javascript
// main.js
const worker = new Worker("worker.js");

worker.postMessage({ data: largeArray });

worker.onmessage = (e) => {
  console.log("Result:", e.data);
};

// worker.js
self.onmessage = (e) => {
  const result = heavyComputation(e.data);
  self.postMessage(result);
};
```

## 💡 检查清单

- [ ] 启用 Gzip/Brotli 压缩
- [ ] 使用 CDN 分发静态资源
- [ ] 图片使用 WebP/AVIF 格式
- [ ] 实现代码分割
- [ ] 关键 CSS 内联
- [ ] 延迟加载非关键资源
- [ ] 使用 HTTP/2
- [ ] 配置合适的缓存策略
- [ ] 移除未使用的 CSS/JS
- [ ] 优化 Web Fonts

## 🔗 相关资源

- [浏览器原理](/docs/frontend/browser/)
- [CSS 动画](/docs/frontend/css/animation)

---

**下一步**：学习 [前端安全](/docs/frontend/advanced/security) 保护应用安全。
