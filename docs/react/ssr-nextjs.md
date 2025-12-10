---
sidebar_position: 24
title: SSR 与 Next.js
---

# 服务端渲染与 Next.js

> [!TIP]
> Next.js 是 React 的全栈框架，支持 SSR、SSG、ISR 等多种渲染方式。

## 📦 安装 Next.js

```bash
npx create-next-app@latest my-app
cd my-app
npm run dev
```

## 🎯 核心概念

### 1. 文件系统路由

```
app/
├── page.tsx          # /
├── about/
│   └── page.tsx      # /about
├── blog/
│   ├── page.tsx      # /blog
│   └── [id]/
│       └── page.tsx  # /blog/:id
```

### 2. 服务端组件（默认）

```tsx
// app/page.tsx - 服务端组件
async function HomePage() {
  const data = await fetch("https://api.example.com/data");
  const posts = await data.json();

  return (
    <div>
      <h1>Posts</h1>
      {posts.map((post) => (
        <div key={post.id}>{post.title}</div>
      ))}
    </div>
  );
}
```

### 3. 客户端组件

```tsx
"use client"; // 标记为客户端组件

import { useState } from "react";

export function Counter() {
  const [count, setCount] = useState(0);

  return <button onClick={() => setCount(count + 1)}>Count: {count}</button>;
}
```

## 🔄 数据获取

### SSG（静态生成）

```tsx
// 构建时获取数据
export async function generateStaticParams() {
  const posts = await fetch("https://...").then((r) => r.json());
  return posts.map((post) => ({ id: post.id.toString() }));
}

async function Post({ params }: { params: { id: string } }) {
  const post = await fetch(`https://.../${params.id}`).then((r) => r.json());
  return <div>{post.title}</div>;
}
```

### SSR（服务端渲染）

```tsx
// 每次请求时获取数据
async function DynamicPage() {
  const data = await fetch("https://...", { cache: "no-store" });
  return <div>{data.title}</div>;
}
```

### ISR（增量静态再生）

```tsx
async function PostsPage() {
  const data = await fetch("https://...", {
    next: { revalidate: 3600 }, // 1小时后重新验证
  });
  return <div>...</div>;
}
```

## 🚀 API 路由

```ts
// app/api/hello/route.ts
export async function GET(request: Request) {
  return Response.json({ message: "Hello" });
}

export async function POST(request: Request) {
  const body = await request.json();
  return Response.json({ received: body });
}
```

---

**了解更多**：查看 [Next.js 官方文档](https://nextjs.org/docs)
