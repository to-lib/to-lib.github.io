---
sidebar_position: 10
title: Context API
---

# Context API

> [!TIP]
> Context 提供了一种在组件树中共享数据的方法，无需通过每一层组件手动传递 props。

## 📚 什么是 Context?

Context 解决了"prop drilling"问题（props 需要层层传递），适用于全局数据如主题、用户信息、语言设置等。

### Props Drilling 问题

```jsx
// ✗ Props Drilling - props 层层传递
function App() {
  const user = { name: "Alice" };
  return <Page user={user} />;
}

function Page({ user }) {
  return <Content user={user} />;
}

function Content({ user }) {
  return <Sidebar user={user} />;
}

function Sidebar({ user }) {
  return <UserInfo user={user} />;
}

function UserInfo({ user }) {
  return <div>{user.name}</div>;
}
```

### 使用 Context 解决

```jsx
// ✓ 使用 Context - 直接访问
const UserContext = createContext();

function App() {
  const user = { name: "Alice" };
  return (
    <UserContext.Provider value={user}>
      <Page />
    </UserContext.Provider>
  );
}

function UserInfo() {
  const user = useContext(UserContext);
  return <div>{user.name}</div>;
}
```

## 🎯 基础用法

### 创建和使用 Context

```jsx
import { createContext, useContext, useState } from "react";

// 1. 创建 Context
const ThemeContext = createContext("light");

// 2. Provider 提供数据
function App() {
  const [theme, setTheme] = useState("light");

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      <Toolbar />
    </ThemeContext.Provider>
  );
}

// 3. Consumer 使用数据
function ThemedButton() {
  const { theme, setTheme } = useContext(ThemeContext);

  return (
    <button
      style={{
        background: theme === "dark" ? "#333" : "#FFF",
        color: theme === "dark" ? "#FFF" : "#333",
      }}
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      Toggle Theme
    </button>
  );
}
```

## 🔧 完整示例

### 主题切换

```jsx
// ThemeContext.js
import { createContext, useContext, useState } from "react";

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState("light");

  const value = {
    theme,
    setTheme,
    toggleTheme: () => setTheme(theme === "light" ? "dark" : "light"),
  };

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return context;
}

// App.js
import { ThemeProvider, useTheme } from "./ThemeContext";

function App() {
  return (
    <ThemeProvider>
      <Page />
    </ThemeProvider>
  );
}

function Page() {
  const { theme } = useTheme();

  return (
    <div className={`app theme-${theme}`}>
      <Header />
      <Content />
    </div>
  );
}

function Header() {
  const { theme, toggleTheme } = useTheme();

  return (
    <header>
      <h1>Current Theme: {theme}</h1>
      <button onClick={toggleTheme}>Toggle Theme</button>
    </header>
  );
}
```

### 用户认证

```jsx
// AuthContext.js
const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const login = async (email, password) => {
    const user = await api.login(email, password);
    setUser(user);
  };

  const logout = () => {
    setUser(null);
  };

  const value = {
    user,
    loading,
    login,
    logout,
    isAuthenticated: !!user,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}

// 使用
function Profile() {
  const { user, logout } = useAuth();

  if (!user) {
    return <div>Please log in</div>;
  }

  return (
    <div>
      <h2>Welcome, {user.name}!</h2>
      <button onClick={logout}>Logout</button>
    </div>
  );
}
```

## 🎨 多个 Context

```jsx
// 组合多个 Context
function App() {
  return (
    <AuthProvider>
      <ThemeProvider>
        <LanguageProvider>
          <Main />
        </LanguageProvider>
      </ThemeProvider>
    </AuthProvider>
  );
}

// 使用多个 Context
function Header() {
  const { user } = useAuth();
  const { theme } = useTheme();
  const { language } = useLanguage();

  return (
    <header className={theme}>
      <span>
        {language === "en" ? "Hello" : "你好"}, {user.name}
      </span>
    </header>
  );
}
```

## ⚡ 性能优化

### 问题：过度渲染

```jsx
// ✗ 问题 - theme 变化会导致所有消费者重新渲染
const ThemeContext = createContext();

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState("light");
  const [user, setUser] = useState(null);

  return (
    <ThemeContext.Provider value={{ theme, setTheme, user, setUser }}>
      {children}
    </ThemeContext.Provider>
  );
}
```

### 解决方案 1：拆分 Context

```jsx
// ✓ 拆分成多个 Context
const ThemeContext = createContext();
const UserContext = createContext();

function Providers({ children }) {
  const [theme, setTheme] = useState("light");
  const [user, setUser] = useState(null);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      <UserContext.Provider value={{ user, setUser }}>
        {children}
      </UserContext.Provider>
    </ThemeContext.Provider>
  );
}
```

### 解决方案 2：useMemo

```jsx
function ThemeProvider({ children }) {
  const [theme, setTheme] = useState("light");

  // 使用 useMemo 缓存 value
  const value = useMemo(() => ({ theme, setTheme }), [theme]);

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}
```

## 💡 最佳实践

### 1. 创建自定义 Hook

```jsx
// ✓ 推荐 - 封装 useContext
const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  // ...
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return context;
}

// 使用
import { useTheme } from "./ThemeContext";

function Component() {
  const { theme } = useTheme(); // 直接使用
}
```

### 2. 提供默认值

```jsx
const ThemeContext = createContext({
  theme: "light",
  toggleTheme: () => {},
});
```

### 3. 拆分状态和更新函数

```jsx
// 拆分为两个 Context
const StateContext = createContext();
const DispatchContext = createContext();

function Provider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  return (
    <DispatchContext.Provider value={dispatch}>
      <StateContext.Provider value={state}>{children}</StateContext.Provider>
    </DispatchContext.Provider>
  );
}

// 只需要状态的组件不会因 dispatch 变化而重新渲染
function useStore() {
  return useContext(StateContext);
}

function useStoreDispatch() {
  return useContext(DispatchContext);
}
```

## 📖 实用示例

### 购物车 Context

```jsx
const CartContext = createContext();

export function CartProvider({ children }) {
  const [items, setItems] = useState([]);

  const addItem = (product) => {
    setItems([...items, { ...product, quantity: 1 }]);
  };

  const removeItem = (id) => {
    setItems(items.filter((item) => item.id !== id));
  };

  const updateQuantity = (id, quantity) => {
    setItems(
      items.map((item) => (item.id === id ? { ...item, quantity } : item))
    );
  };

  const total = items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );

  const value = {
    items,
    addItem,
    removeItem,
    updateQuantity,
    total,
    itemCount: items.length,
  };

  return <CartContext.Provider value={value}>{children}</CartContext.Provider>;
}

export function useCart() {
  return useContext(CartContext);
}

// 使用
function CartButton() {
  const { itemCount } = useCart();
  return <button>Cart ({itemCount})</button>;
}
```

### 通知系统

```jsx
const NotificationContext = createContext();

export function NotificationProvider({ children }) {
  const [notifications, setNotifications] = useState([]);

  const addNotification = (message, type = "info") => {
    const id = Date.now();
    setNotifications([...notifications, { id, message, type }]);

    // 3秒后自动移除
    setTimeout(() => {
      removeNotification(id);
    }, 3000);
  };

  const removeNotification = (id) => {
    setNotifications(notifications.filter((n) => n.id !== id));
  };

  return (
    <NotificationContext.Provider value={{ addNotification }}>
      {children}
      <div className="notifications">
        {notifications.map((notif) => (
          <div key={notif.id} className={`notification ${notif.type}`}>
            {notif.message}
            <button onClick={() => removeNotification(notif.id)}>×</button>
          </div>
        ))}
      </div>
    </NotificationContext.Provider>
  );
}

export function useNotification() {
  return useContext(NotificationContext);
}
```

---

**下一步**: 学习 [性能优化](./performance-optimization) 提升应用性能，或查看 [状态管理](./state-management) 了解更多状态管理方案。
