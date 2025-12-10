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

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<div>Loading...</div>}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
```

## 📦 动态导入

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

## 💡 最佳实践

```jsx
// ✓ 好：按路由分割
const Dashboard = lazy(() => import("./Dashboard"));

// ✓ 好：按功能分割
const Chart = lazy(() => import("./Chart"));

// ✗ 不好：过度分割（太小的组件）
const Button = lazy(() => import("./Button")); // 不推荐
```

---

**相关主题**：[性能优化](./performance-optimization)
