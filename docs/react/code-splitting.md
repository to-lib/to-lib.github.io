---
sidebar_position: 25
title: 代码分割
---

# 代码分割与懒加载

> [!TIP]
> 代码分割可以显著提升应用加载性能。本文介绍 React.lazy 和动态导入。

## 🚀 React.lazy

### 基础用法

```jsx
import { lazy, Suspense } from "react";

// 懒加载组件
const HeavyComponent = lazy(() => import("./HeavyComponent"));

function App() {
  return (
    <Suspense fallback={<div>加载中...</div>}>
      <HeavyComponent />
    </Suspense>
  );
}
```

### 路由级代码分割

```jsx
import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

const Home = lazy(() => import("./pages/Home"));
const About = lazy(() => import("./pages/About"));
const Contact = lazy(() => import("./pages/Contact"));
const Dashboard = lazy(() => import("./pages/Dashboard"));

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<LoadingSpinner />}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/dashboard/*" element={<Dashboard />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
```

## 📦 动态导入

### 按需加载组件

```jsx
import { useState } from "react";

function App() {
  const [Component, setComponent] = useState(null);

  const loadComponent = async () => {
    const module = await import("./HeavyComponent");
    setComponent(() => module.default);
  };

  return (
    <div>
      <button onClick={loadComponent}>加载组件</button>
      {Component && <Component />}
    </div>
  );
}
```

### 按需加载模块

```jsx
async function processData(data) {
  // 只在需要时加载大型库
  const { processLargeData } = await import("./heavyProcessing");
  return processLargeData(data);
}

// 按需加载工具库
async function formatDate(date) {
  const { format } = await import("date-fns");
  return format(date, "yyyy-MM-dd");
}
```

## ⚡ 预加载策略

### 鼠标悬停预加载

```jsx
const Dashboard = lazy(() => import("./pages/Dashboard"));

// 预加载函数
const preloadDashboard = () => {
  import("./pages/Dashboard");
};

function Navigation() {
  return (
    <nav>
      <Link
        to="/dashboard"
        onMouseEnter={preloadDashboard}
        onFocus={preloadDashboard}
      >
        仪表盘
      </Link>
    </nav>
  );
}
```

### 页面可见时预加载

```jsx
import { useEffect, useRef } from "react";

function PreloadOnVisible({ preload, children }) {
  const ref = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          preload();
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, [preload]);

  return <div ref={ref}>{children}</div>;
}

// 使用
<PreloadOnVisible preload={() => import("./HeavyChart")}>
  <AreaWhereChartWillBeLoaded />
</PreloadOnVisible>;
```

### 空闲时预加载

```jsx
function useIdlePreload(modules) {
  useEffect(() => {
    if ("requestIdleCallback" in window) {
      const id = requestIdleCallback(() => {
        modules.forEach((mod) => mod());
      });
      return () => cancelIdleCallback(id);
    } else {
      // 降级方案
      const id = setTimeout(() => {
        modules.forEach((mod) => mod());
      }, 2000);
      return () => clearTimeout(id);
    }
  }, [modules]);
}

// 使用
useIdlePreload([
  () => import("./pages/Settings"),
  () => import("./pages/Profile"),
]);
```

## 🛡️ 错误边界结合

```jsx
import { lazy, Suspense, Component } from "react";

// 错误边界组件
class ErrorBoundary extends Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  retry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-container">
          <h2>加载失败</h2>
          <p>{this.state.error?.message}</p>
          <button onClick={this.retry}>重试</button>
        </div>
      );
    }
    return this.props.children;
  }
}

// 结合 Suspense 使用
const AsyncComponent = lazy(() => import("./AsyncComponent"));

function App() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<LoadingSpinner />}>
        <AsyncComponent />
      </Suspense>
    </ErrorBoundary>
  );
}
```

### 带重试的懒加载

```jsx
function lazyWithRetry(importFn, retries = 3) {
  return lazy(async () => {
    for (let i = 0; i < retries; i++) {
      try {
        return await importFn();
      } catch (error) {
        if (i === retries - 1) throw error;
        // 等待后重试
        await new Promise((r) => setTimeout(r, 1000 * (i + 1)));
      }
    }
  });
}

// 使用
const Dashboard = lazyWithRetry(() => import("./pages/Dashboard"));
```

## 🎨 加载状态 UI

### Skeleton 骨架屏

```jsx
function PageSkeleton() {
  return (
    <div className="skeleton-container">
      <div className="skeleton-header" />
      <div className="skeleton-nav" />
      <div className="skeleton-content">
        {[1, 2, 3].map((i) => (
          <div key={i} className="skeleton-item" />
        ))}
      </div>
    </div>
  );
}

const Dashboard = lazy(() => import("./Dashboard"));

function App() {
  return (
    <Suspense fallback={<PageSkeleton />}>
      <Dashboard />
    </Suspense>
  );
}
```

### 带进度条的加载

```jsx
import { useState, useEffect } from "react";

function LoadingProgress({ isLoading }) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (isLoading) {
      const interval = setInterval(() => {
        setProgress((p) => Math.min(p + 10, 90));
      }, 100);
      return () => clearInterval(interval);
    } else {
      setProgress(100);
    }
  }, [isLoading]);

  if (!isLoading && progress === 100) return null;

  return (
    <div className="progress-bar">
      <div className="progress" style={{ width: `${progress}%` }} />
    </div>
  );
}
```

## 🔧 构建配置优化

### Vite 配置

```js
// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // 手动分割代码块
        manualChunks: {
          // 第三方库单独打包
          vendor: ["react", "react-dom", "react-router-dom"],
          // UI 组件库
          ui: ["@radix-ui/react-dialog", "@radix-ui/react-dropdown-menu"],
          // 图表库
          charts: ["recharts", "d3"],
        },
      },
    },
    // 分块大小警告限制
    chunkSizeWarningLimit: 500,
  },
});
```

### Webpack 配置

```js
// webpack.config.js
module.exports = {
  optimization: {
    splitChunks: {
      chunks: "all",
      cacheGroups: {
        vendors: {
          test: /[\\/]node_modules[\\/]/,
          name: "vendors",
          priority: 10,
        },
        react: {
          test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
          name: "react",
          priority: 20,
        },
        common: {
          minChunks: 2,
          priority: 5,
          reuseExistingChunk: true,
        },
      },
    },
  },
};
```

## 📊 性能监控

### 分析包大小

```bash
# Vite
npx vite-bundle-visualizer

# Webpack
npx webpack-bundle-analyzer stats.json
```

### 监控加载性能

```jsx
// 使用 Web Vitals
import { onLCP, onFCP, onTTFB } from "web-vitals";

function reportWebVitals() {
  onLCP(console.log); // 最大内容绘制
  onFCP(console.log); // 首次内容绘制
  onTTFB(console.log); // 首字节时间
}

// 监控懒加载时间
async function measureLazyLoad(name, importFn) {
  const start = performance.now();
  const module = await importFn();
  const duration = performance.now() - start;

  console.log(`${name} 加载耗时: ${duration.toFixed(2)}ms`);

  // 上报到监控服务
  if (duration > 1000) {
    reportSlowLoad(name, duration);
  }

  return module;
}
```

## 💡 最佳实践

```jsx
// ✓ 好：按路由分割
const Dashboard = lazy(() => import("./Dashboard"));

// ✓ 好：按功能分割
const Chart = lazy(() => import("./Chart"));

// ✓ 好：大型第三方库动态导入
const Editor = lazy(() => import("./Monaco-Editor"));

// ✗ 不好：过度分割（太小的组件）
const Button = lazy(() => import("./Button")); // 不推荐

// ✗ 不好：关键路径组件懒加载
const Header = lazy(() => import("./Header")); // 不推荐
```

| 场景          | 推荐做法     |
| ------------- | ------------ |
| 路由页面      | ✓ 懒加载     |
| 大型第三方库  | ✓ 动态导入   |
| 模态框/对话框 | ✓ 懒加载     |
| 导航/Header   | ✗ 不要懒加载 |
| 小型组件      | ✗ 不要懒加载 |

---

**相关主题**：[性能优化](/docs/react/performance-optimization) | [最佳实践](/docs/react/best-practices)
