---
sidebar_position: 21
title: 状态管理
---

# React 状态管理

> [!TIP]
> 状态管理是 React 应用的核心。本文对比多种状态管理方案，帮助你选择最适合的解决方案。

## 📊 状态管理方案对比

| 方案         | 适用场景     | 学习曲线 | 性能 | 推荐度     |
| ------------ | ------------ | -------- | ---- | ---------- |
| **useState** | 组件内部状态 | 低       | 高   | ⭐⭐⭐⭐⭐ |
| **Context**  | 中小型应用   | 低       | 中   | ⭐⭐⭐⭐   |
| **Zustand**  | 中大型应用   | 低       | 高   | ⭐⭐⭐⭐⭐ |
| **Redux**    | 大型应用     | 高       | 中   | ⭐⭐⭐     |
| **Jotai**    | 原子化状态   | 中       | 高   | ⭐⭐⭐⭐   |
| **Recoil**   | 复杂状态图   | 中       | 高   | ⭐⭐⭐     |

## 🎯 Context API

### 基础用法

```jsx
import { createContext, useContext, useState } from "react";

// 创建 Context
const ThemeContext = createContext();

// Provider 组件
function ThemeProvider({ children }) {
  const [theme, setTheme] = useState("light");

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// 使用
function App() {
  return (
    <ThemeProvider>
      <Header />
      <Main />
    </ThemeProvider>
  );
}

function Header() {
  const { theme, toggleTheme } = useContext(ThemeContext);

  return (
    <header className={theme}>
      <button onClick={toggleTheme}>切换主题</button>
    </header>
  );
}
```

## 🚀 Zustand（推荐）

### 安装

```bash
npm install zustand
```

### 基础用法

```jsx
import { create } from "zustand";

// 创建 store
const useStore = create((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
  decrement: () => set((state) => ({ count: state.count - 1 })),
  reset: () => set({ count: 0 }),
}));

// 使用
function Counter() {
  const { count, increment, decrement, reset } = useStore();

  return (
    <div>
      <h1>{count}</h1>
      <button onClick={increment}>+</button>
      <button onClick={decrement}>-</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
}
```

### 高级用法

```jsx
const useUserStore = create((set, get) => ({
  user: null,
  loading: false,
  error: null,

  // 异步 action
  fetchUser: async (id) => {
    set({ loading: true, error: null });
    try {
      const res = await fetch(`/api/users/${id}`);
      const user = await res.json();
      set({ user, loading: false });
    } catch (error) {
      set({ error: error.message, loading: false });
    }
  },

  // 访问其他 slice
  updateProfile: (data) => {
    const currentUser = get().user;
    set({ user: { ...currentUser, ...data } });
  },
}));
```

## 🔧 Redux Toolkit（传统大型应用）

### 安装

```bash
npm install @reduxjs/toolkit react-redux
```

### 基础配置

```jsx
import { configureStore, createSlice } from "@reduxjs/toolkit";
import { Provider, useSelector, useDispatch } from "react-redux";

// 创建 slice
const counterSlice = createSlice({
  name: "counter",
  initialState: { value: 0 },
  reducers: {
    increment: (state) => {
      state.value += 1;
    },
    decrement: (state) => {
      state.value -= 1;
    },
    incrementByAmount: (state, action) => {
      state.value += action.payload;
    },
  },
});

export const { increment, decrement, incrementByAmount } = counterSlice.actions;

// 配置 store
const store = configureStore({
  reducer: {
    counter: counterSlice.reducer,
  },
});

// 提供 store
function App() {
  return (
    <Provider store={store}>
      <Counter />
    </Provider>
  );
}

// 使用
function Counter() {
  const count = useSelector((state) => state.counter.value);
  const dispatch = useDispatch();

  return (
    <div>
      <h1>{count}</h1>
      <button onClick={() => dispatch(increment())}>+</button>
      <button onClick={() => dispatch(decrement())}>-</button>
      <button onClick={() => dispatch(incrementByAmount(5))}>+5</button>
    </div>
  );
}
```

## ⚛️ Jotai（原子化状态）

### 安装

```bash
npm install jotai
```

### 基础用法

```jsx
import { atom, useAtom } from "jotai";

// 创建原子
const countAtom = atom(0);
const doubleAtom = atom((get) => get(countAtom) * 2);

// 使用
function Counter() {
  const [count, setCount] = useAtom(countAtom);
  const [double] = useAtom(doubleAtom);

  return (
    <div>
      <p>Count: {count}</p>
      <p>Double: {double}</p>
      <button onClick={() => setCount((c) => c + 1)}>+</button>
    </div>
  );
}
```

## 💡 最佳实践

### 1. 选择合适的方案

```jsx
// ✓ 简单状态 - useState
function Component() {
  const [open, setOpen] = useState(false);
  return <Modal open={open} onClose={() => setOpen(false)} />;
}

// ✓ 中小型应用 - Context
<ThemeProvider>
  <App />
</ThemeProvider>;

// ✓ 大型应用 - Zustand/Redux
const useAppStore = create((set) => ({
  // 全局状态
}));
```

### 2. 避免过度使用全局状态

```jsx
// ✗ 不好：所有状态都放全局
const useStore = create(() => ({
  modalOpen: false,
  inputValue: "",
  // ...太多局部状态
}));

// ✓ 好：只把真正全局的状态放全局
const useAuthStore = create(() => ({
  user: null,
  token: null,
}));

// 局部状态用 useState
function Modal() {
  const [open, setOpen] = useState(false);
  // ...
}
```

---

**下一步**：学习 [TypeScript](./typescript) 增强类型安全，或查看 [测试](./testing) 保证代码质量。
