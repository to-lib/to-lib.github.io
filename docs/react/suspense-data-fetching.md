---
sidebar_position: 16
title: Suspense 与 use() 数据获取
---

# Suspense 与 use() 数据获取

> [!TIP]
> `Suspense` 不仅能用于代码分割，也可以统一管理“数据加载中的 UI”。
> React 19 提供了 `use()` 来读取 Promise（以及 Context），从而更自然地配合 `Suspense`。

## 🧠 两类 Suspense：代码分割 vs 数据获取

- **代码分割**：`React.lazy(() => import(...))` + `Suspense`（加载组件代码）
- **数据获取**：组件在渲染时“读取数据”，如果数据未就绪则“挂起”，交给最近的 `Suspense` fallback

代码分割示例可参考：

- [代码分割](/docs/react/code-splitting)

## 🎣 use() 读取 Promise

React 19 中，你可以通过 `use(promise)` 直接读取 Promise 的结果。

```jsx
import { Suspense, use } from "react";

function fetchUser(userId) {
  return fetch(`/api/users/${userId}`).then((r) => r.json());
}

function UserProfile({ userPromise }) {
  const user = use(userPromise);
  return <div>{user.name}</div>;
}

export function App({ userId }) {
  const userPromise = fetchUser(userId);

  return (
    <Suspense fallback={<div>Loading user...</div>}>
      <UserProfile userPromise={userPromise} />
    </Suspense>
  );
}
```

## ⚠️ 关键注意点：不要在每次渲染都创建新 Promise

如果你在组件渲染时每次都 `fetch()`，会导致：

- promise 引用变化 -> 反复挂起
- 请求被重复触发

常见做法是把 promise “缓存”起来（示意）：

```jsx
const cache = new Map();

function fetchUserCached(userId) {
  if (!cache.has(userId)) {
    cache.set(userId, fetch(`/api/users/${userId}`).then((r) => r.json()));
  }
  return cache.get(userId);
}

function UserProfile({ userId }) {
  const user = use(fetchUserCached(userId));
  return <div>{user.name}</div>;
}
```

在真实项目中，更推荐使用成熟的缓存方案（例如 TanStack Query）：

- [数据获取（TanStack Query）](/docs/react/data-fetching)

## 🧯 错误处理：配合 Error Boundary

数据读取失败时，你通常希望在 UI 层兜底。

- 推荐阅读： [错误边界](/docs/react/error-boundaries)

> 实战中常见结构：`ErrorBoundary` 包 `Suspense`，并为不同区域提供不同 fallback。

## ✅ 什么时候适合用这种模式

- 你希望页面“按区块逐步展示”（streaming / progressive rendering）
- loading UI 想统一由 `Suspense` 控制，而不是每个组件里写 `isLoading`
- 你在用 Next.js App Router / Server Components（更容易天然配合）

## 🔗 相关资源

- [React 19 新特性](/docs/react/react19-features)
- [Hooks 详解](/docs/react/hooks)
