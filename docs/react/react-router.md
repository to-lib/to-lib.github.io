---
sidebar_position: 20
title: React Router
---

# React Router 路由管理

> [!TIP]
> React Router 是 React 应用中最流行的路由解决方案。本文基于 React Router v6，涵盖基础到高级用法。

## 📦 安装

```bash
npm install react-router-dom
```

## 🚀 快速开始

### 基础路由配置

```jsx
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";

function App() {
  return (
    <BrowserRouter>
      <nav>
        <Link to="/">首页</Link>
        <Link to="/about">关于</Link>
        <Link to="/contact">联系</Link>
      </nav>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />
      </Routes>
    </BrowserRouter>
  );
}
```

## 🎯 核心概念

### 1. 路由参数

```jsx
import { useParams } from "react-router-dom";

function App() {
  return (
    <Routes>
      <Route path="/users/:id" element={<UserProfile />} />
      <Route path="/posts/:postId/comments/:commentId" element={<Comment />} />
    </Routes>
  );
}

function UserProfile() {
  const { id } = useParams();
  return <div>用户 ID: {id}</div>;
}
```

### 2. 查询参数

```jsx
import { useSearchParams } from "react-router-dom";

function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const query = searchParams.get("q");
  const page = searchParams.get("page") || "1";

  const handleSearch = (newQuery) => {
    setSearchParams({ q: newQuery, page: "1" });
  };

  return (
    <div>
      <p>搜索: {query}</p>
      <p>页码: {page}</p>
    </div>
  );
}
```

### 3. 嵌套路由

```jsx
function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="about" element={<About />} />
        <Route path="dashboard" element={<Dashboard />}>
          <Route index element={<Overview />} />
          <Route path="settings" element={<Settings />} />
          <Route path="profile" element={<Profile />} />
        </Route>
      </Route>
    </Routes>
  );
}

function Layout() {
  return (
    <div>
      <nav>{/* 导航栏 */}</nav>
      <Outlet /> {/* 渲染子路由 */}
    </div>
  );
}

function Dashboard() {
  return (
    <div>
      <aside>{/* 侧边栏 */}</aside>
      <main>
        <Outlet /> {/* 渲染嵌套路由 */}
      </main>
    </div>
  );
}
```

### 4. 编程式导航

```jsx
import { useNavigate } from "react-router-dom";

function LoginForm() {
  const navigate = useNavigate();

  const handleSubmit = async (credentials) => {
    const success = await login(credentials);
    if (success) {
      navigate("/dashboard"); // 跳转
      // navigate(-1); // 返回上一页
      // navigate('/home', { replace: true }); // 替换历史记录
    }
  };

  return <form onSubmit={handleSubmit}>...</form>;
}
```

## 🔒 受保护的路由

```jsx
function ProtectedRoute({ children }) {
  const { user } = useAuth();

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

// 使用
<Route
  path="/dashboard"
  element={
    <ProtectedRoute>
      <Dashboard />
    </ProtectedRoute>
  }
/>;
```

## 📊 完整示例

```jsx
import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  NavLink,
  Navigate,
  Outlet,
  useParams,
  useNavigate,
  useLocation,
} from "react-router-dom";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="products" element={<Products />} />
          <Route path="products/:id" element={<ProductDetail />} />
          <Route path="cart" element={<Cart />} />
          <Route path="login" element={<Login />} />
          <Route
            path="account"
            element={
              <ProtectedRoute>
                <Account />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

function Layout() {
  return (
    <div>
      <header>
        <nav>
          <NavLink to="/">首页</NavLink>
          <NavLink to="/products">产品</NavLink>
          <NavLink to="/cart">购物车</NavLink>
          <NavLink to="/account">账户</NavLink>
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
      <footer>© 2024</footer>
    </div>
  );
}
```

---

**下一步**：学习 [状态管理](./state-management) 管理全局状态，或查看 [TypeScript](./typescript) 提升代码质量。
