---
sidebar_position: 15
title: 数据获取（TanStack Query）
---

# 数据获取（TanStack Query / React Query）

> [!TIP]
> 在 React 应用里，“数据获取”属于 **Server State**：它通常来自远端、会失效、需要缓存与重试。
> TanStack Query（常被称为 React Query）是目前最常用的 Server State 管理方案之一。

## 🎯 为什么不建议用 useEffect + fetch 堆起来

用 `useEffect` 直接请求接口很快会遇到：

- 重复请求、缓存难
- loading/error 状态重复写
- 并发请求、取消、重试、失效策略难
- 需要手动维护“哪些数据要重新拉取”

TanStack Query 把这些能力做成了统一抽象：

- **Query Key**：用 key 作为缓存索引
- **Cache**：缓存、失效、过期
- **Retry**：失败重试、退避
- **DevTools**：调试缓存/请求状态

## 📦 安装（在你的业务项目中）

```bash
pnpm add @tanstack/react-query
```

## 🧱 基础接入

```jsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <YourRoutes />
    </QueryClientProvider>
  );
}
```

## 🔍 useQuery：读取数据

```jsx
import { useQuery } from "@tanstack/react-query";

function fetchUsers() {
  return fetch("/api/users").then((r) => r.json());
}

export function UserList() {
  const { data, isPending, error } = useQuery({
    queryKey: ["users"],
    queryFn: fetchUsers,
  });

  if (isPending) return <div>Loading...</div>;
  if (error) return <div>Error</div>;

  return (
    <ul>
      {data.map((u) => (
        <li key={u.id}>{u.name}</li>
      ))}
    </ul>
  );
}
```

## ✍️ useMutation：提交变更

```jsx
import { useMutation, useQueryClient } from "@tanstack/react-query";

function createUser(payload) {
  return fetch("/api/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then((r) => r.json());
}

export function CreateUserForm() {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: createUser,
    onSuccess: () => {
      // 让 users 缓存失效，触发重新拉取
      queryClient.invalidateQueries({ queryKey: ["users"] });
    },
  });

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        const formData = new FormData(e.currentTarget);
        mutation.mutate({ name: String(formData.get("name")) });
      }}
    >
      <input name="name" placeholder="name" />
      <button disabled={mutation.isPending}>
        {mutation.isPending ? "Saving..." : "Create"}
      </button>
      {mutation.error && <div>Submit failed</div>}
    </form>
  );
}
```

## ⚡ 与 React 19：Suspense / streaming

如果你希望把 loading UI 交给 `Suspense` 统一管理，可以进一步学习：

- [Suspense 与 use() 数据获取](/docs/react/suspense-data-fetching)

## ✅ 最佳实践

- **Query Key 设计**：把影响结果的参数都放进 key（如 `["users", page, pageSize]`）
- **请求函数稳定**：保证 `queryFn` 行为可预测（同 key 同结果）
- **失效而不是手动 setState**：对“远端数据”优先用 `invalidateQueries`

## 🔗 相关资源

- [TanStack Query 文档](https://tanstack.com/query/latest)
- [状态管理](/docs/react/state-management)
- [React 19 新特性](/docs/react/react19-features)
