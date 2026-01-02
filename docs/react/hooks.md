---
sidebar_position: 6
title: Hooks 详解
---

# React Hooks 详解

> [!TIP]
> Hooks 是 React 16.8 引入的革命性特性，让函数组件也能使用 state 和生命周期等功能。React 19 进一步增强了 Hooks 的能力。

## 📚 什么是 Hooks？

Hooks 是特殊的函数，让你在函数组件中"钩入" React 特性。

### Hooks 规则

1. ✅ 只在函数组件或自定义 Hook 中调用
2. ✅ 只在函数顶层调用，不要在循环、条件或嵌套函数中调用
3. ✅ Hook 的调用顺序必须保持一致

## useState - 状态管理

### 基础用法

```jsx
import { useState } from "react";

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
    </div>
  );
}
```

### 多个状态

```jsx
function UserProfile() {
  const [name, setName] = useState("");
  const [age, setAge] = useState(0);
  const [email, setEmail] = useState("");

  return (
    <form>
      <input value={name} onChange={(e) => setName(e.target.value)} />
      <input
        type="number"
        value={age}
        onChange={(e) => setAge(e.target.value)}
      />
      <input value={email} onChange={(e) => setEmail(e.target.value)} />
    </form>
  );
}
```

### 对象状态

```jsx
function Form() {
  const [formData, setFormData] = useState({
    username: "",
    password: "",
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  return (
    <form>
      <input
        name="username"
        value={formData.username}
        onChange={handleChange}
      />
      <input
        name="password"
        type="password"
        value={formData.password}
        onChange={handleChange}
      />
    </form>
  );
}
```

### 函数式更新

```jsx
function Counter() {
  const [count, setCount] = useState(0);

  const increment = () => {
    // ✗ 不推荐：基于当前state
    setCount(count + 1);

    // ✓ 推荐：使用函数式更新
    setCount((prevCount) => prevCount + 1);
  };

  return <button onClick={increment}>Count: {count}</button>;
}
```

## ⚡ useEffect - 副作用处理

### 基础用法

```jsx
import { useState, useEffect } from "react";

function App() {
  const [count, setCount] = useState(0);

  // 每次渲染后执行
  useEffect(() => {
    document.title = `You clicked ${count} times`;
  });

  return <button onClick={() => setCount(count + 1)}>Click me</button>;
}
```

### 依赖数组

```jsx
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    // 只在 userId 变化时执行
    fetch(`/api/users/${userId}`)
      .then((res) => res.json())
      .then((data) => setUser(data));
  }, [userId]); // 依赖数组

  return user ? <div>{user.name}</div> : <div>Loading...</div>;
}
```

### 清理函数

```jsx
function Timer() {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setSeconds((s) => s + 1);
    }, 1000);

    // 清理函数：组件卸载时执行
    return () => clearInterval(interval);
  }, []);

  return <div>Seconds: {seconds}</div>;
}
```

### 常见场景

```jsx
function Component() {
  // 只在挂载时执行一次
  useEffect(() => {
    console.log("Component mounted");
    return () => console.log("Component unmounted");
  }, []);

  // 监听多个依赖
  useEffect(() => {
    console.log("prop1 or prop2 changed");
  }, [prop1, prop2]);

  // 每次渲染都执行（谨慎使用）
  useEffect(() => {
    console.log("Component rendered");
  });
}
```

## 🎣 useContext - 跨组件共享状态

```jsx
import { createContext, useContext, useState } from "react";

// 创建 Context
const ThemeContext = createContext();

function App() {
  const [theme, setTheme] = useState("light");

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      <Toolbar />
    </ThemeContext.Provider>
  );
}

function Toolbar() {
  return (
    <div>
      <ThemedButton />
    </div>
  );
}

function ThemedButton() {
  // 使用 Context
  const { theme, setTheme } = useContext(ThemeContext);

  return (
    <button
      style={{
        background: theme === "dark" ? "#333" : "#FFF",
        color: theme === "dark" ? "#FFF" : "#333",
      }}
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      Toggle Theme (Current: {theme})
    </button>
  );
}
```

## 🚀 useReducer - 复杂状态管理

```jsx
import { useReducer } from "react";

// 定义 reducer
function counterReducer(state, action) {
  switch (action.type) {
    case "increment":
      return { count: state.count + 1 };
    case "decrement":
      return { count: state.count - 1 };
    case "reset":
      return { count: 0 };
    default:
      throw new Error(`Unknown action: ${action.type}`);
  }
}

function Counter() {
  const [state, dispatch] = useReducer(counterReducer, { count: 0 });

  return (
    <div>
      <p>Count: {state.count}</p>
      <button onClick={() => dispatch({ type: "increment" })}>+1</button>
      <button onClick={() => dispatch({ type: "decrement" })}>-1</button>
      <button onClick={() => dispatch({ type: "reset" })}>Reset</button>
    </div>
  );
}
```

### 复杂示例：购物车

```jsx
const cartReducer = (state, action) => {
  switch (action.type) {
    case "ADD_ITEM":
      return {
        ...state,
        items: [...state.items, action.payload],
      };
    case "REMOVE_ITEM":
      return {
        ...state,
        items: state.items.filter((item) => item.id !== action.payload),
      };
    case "UPDATE_QUANTITY":
      return {
        ...state,
        items: state.items.map((item) =>
          item.id === action.payload.id
            ? { ...item, quantity: action.payload.quantity }
            : item
        ),
      };
    default:
      return state;
  }
};

function ShoppingCart() {
  const [cart, dispatch] = useReducer(cartReducer, { items: [] });

  const addItem = (product) => {
    dispatch({ type: "ADD_ITEM", payload: product });
  };

  return (
    <div>
      {cart.items.map((item) => (
        <div key={item.id}>
          {item.name} x {item.quantity}
          <button
            onClick={() =>
              dispatch({
                type: "REMOVE_ITEM",
                payload: item.id,
              })
            }
          >
            Remove
          </button>
        </div>
      ))}
    </div>
  );
}
```

## ⚡ useMemo - 性能优化

### 缓存计算结果

```jsx
import { useState, useMemo } from "react";

function ExpensiveComponent({ items }) {
  const [filter, setFilter] = useState("");

  // 只在 items 或 filter 改变时重新计算
  const filteredItems = useMemo(() => {
    console.log("Filtering...");
    return items.filter((item) =>
      item.name.toLowerCase().includes(filter.toLowerCase())
    );
  }, [items, filter]);

  return (
    <div>
      <input value={filter} onChange={(e) => setFilter(e.target.value)} />
      {filteredItems.map((item) => (
        <div key={item.id}>{item.name}</div>
      ))}
    </div>
  );
}
```

## 🔄 useCallback - 缓存函数

```jsx
import { useState, useCallback } from "react";

function Parent() {
  const [count, setCount] = useState(0);

  // 缓存函数，避免每次渲染都创建新函数
  const handleClick = useCallback(() => {
    console.log("Button clicked");
  }, []); // 空依赖数组，函数永远不变

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <Child onClick={handleClick} />
    </div>
  );
}

// 使用 memo 优化子组件
const Child = React.memo(({ onClick }) => {
  console.log("Child rendered");
  return <button onClick={onClick}>Click Me</button>;
});
```

## 🎯 useRef - 引用 DOM 和保存值

### 访问 DOM

```jsx
import { useRef } from "react";

function TextInput() {
  const inputRef = useRef(null);

  const focusInput = () => {
    inputRef.current.focus();
  };

  return (
    <div>
      <input ref={inputRef} />
      <button onClick={focusInput}>Focus Input</button>
    </div>
  );
}
```

### 保存可变值

```jsx
function Timer() {
  const [seconds, setSeconds] = useState(0);
  const intervalRef = useRef(null);

  const start = () => {
    if (!intervalRef.current) {
      intervalRef.current = setInterval(() => {
        setSeconds((s) => s + 1);
      }, 1000);
    }
  };

  const stop = () => {
    clearInterval(intervalRef.current);
    intervalRef.current = null;
  };

  return (
    <div>
      <p>Seconds: {seconds}</p>
      <button onClick={start}>Start</button>
      <button onClick={stop}>Stop</button>
    </div>
  );
}
```

## 🆕 React 19 新 Hooks

### use() - 读取 Promise 和 Context

```jsx
import { use } from "react";

function UserProfile({ userPromise }) {
  // 直接读取 Promise
  const user = use(userPromise);

  return <div>{user.name}</div>;
}

// 或读取 Context
function ThemedComponent() {
  const theme = use(ThemeContext);
  return <div>Theme: {theme}</div>;
}
```

### useFormStatus - 表单状态

````jsx
import { useFormStatus } from "react-dom";

function SubmitButton() {
  const { pending } = useFormStatus();

  return (
    <button type="submit" disabled={pending}>
      {pending ? "Submitting..." : "Submit"}
    </button>
  );
}
````

### useOptimistic - 乐观更新

```jsx
import { useOptimistic } from "react";

function TodoList({ todos }) {
  const [optimisticTodos, addOptimisticTodo] = useOptimistic(
    todos,
    (state, newTodo) => [...state, newTodo]
  );

  async function addTodo(formData) {
    const newTodo = { id: Date.now(), text: formData.get("text") };
    addOptimisticTodo(newTodo); // 立即显示

    await saveTodo(newTodo); // 后台保存
  }

  return (
    <form action={addTodo}>
      <input name="text" />
      <button type="submit">Add</button>
      <ul>
        {optimisticTodos.map((todo) => (
          <li key={todo.id}>{todo.text}</li>
        ))}
      </ul>
    </form>
  );
}
```

### useActionState - 管理 Action 状态

```jsx
import { useActionState } from "react";

function TodoForm() {
  async function createTodo(prevState, formData) {
    const title = formData.get("title");

    if (!title) {
      return { error: "请输入标题" };
    }

    await saveTodo({ title });
    return { success: true };
  }

  const [state, formAction, isPending] = useActionState(createTodo, {});

  return (
    <form action={formAction}>
      <input name="title" placeholder="新待办事项" />
      <button type="submit" disabled={isPending}>
        {isPending ? "添加中..." : "添加"}
      </button>
      {state.error && <p style={{ color: "red" }}>{state.error}</p>}
    </form>
  );
}
```

### useTransition - 标记非紧急更新

```jsx
import { useState, useTransition } from "react";

function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [isPending, startTransition] = useTransition();

  function handleChange(e) {
    const value = e.target.value;
    setQuery(value); // 紧急更新

    startTransition(() => {
      // 非紧急更新，可被中断
      setResults(filterLargeList(value));
    });
  }

  return (
    <div>
      <input value={query} onChange={handleChange} />
      {isPending && <span>搜索中...</span>}
      <ul style={{ opacity: isPending ? 0.7 : 1 }}>
        {results.map((item) => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
}
```

### useDeferredValue - 延迟值更新

```jsx
import { useState, useDeferredValue, useMemo } from "react";

function SearchResults({ query }) {
  const deferredQuery = useDeferredValue(query);
  const isStale = query !== deferredQuery;

  const results = useMemo(() => searchDatabase(deferredQuery), [deferredQuery]);

  return (
    <div style={{ opacity: isStale ? 0.7 : 1 }}>
      {results.map((item) => (
        <div key={item.id}>{item.title}</div>
      ))}
    </div>
  );
}
```

### useId - 生成唯一 ID

```jsx
import { useId } from "react";

function FormField({ label }) {
  const id = useId();

  return (
    <div>
      <label htmlFor={id}>{label}</label>
      <input id={id} type="text" />
    </div>
  );
}

// 多个相关 ID
function PasswordField() {
  const id = useId();

  return (
    <div>
      <label htmlFor={`${id}-password`}>密码</label>
      <input
        id={`${id}-password`}
        type="password"
        aria-describedby={`${id}-hint`}
      />
      <p id={`${id}-hint`}>密码至少 8 个字符</p>
    </div>
  );
}
```

## 🛠️ 自定义 Hook

### 基础示例

```jsx
// useCounter.js
function useCounter(initialValue = 0) {
  const [count, setCount] = useState(initialValue);

  const increment = () => setCount((c) => c + 1);
  const decrement = () => setCount((c) => c - 1);
  const reset = () => setCount(initialValue);

  return { count, increment, decrement, reset };
}

// 使用
function Counter() {
  const { count, increment, decrement, reset } = useCounter(0);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>+</button>
      <button onClick={decrement}>-</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
}
```

### 实用 Hook：useLocalStorage

```jsx
function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      return initialValue;
    }
  });

  const setStoredValue = (newValue) => {
    try {
      setValue(newValue);
      window.localStorage.setItem(key, JSON.stringify(newValue));
    } catch (error) {
      console.error(error);
    }
  };

  return [value, setStoredValue];
}

// 使用
function App() {
  const [name, setName] = useLocalStorage("name", "");

  return <input value={name} onChange={(e) => setName(e.target.value)} />;
}
```

### 实用 Hook：useFetch

```jsx
function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        setError(null);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [url]);

  return { data, loading, error };
}

// 使用
function UserList() {
  const { data: users, loading, error } = useFetch("/api/users");

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <ul>
      {users.map((user) => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

## 📊 Hooks 对比表

| Hook           | 用途                 | 返回值                            |
| -------------- | -------------------- | --------------------------------- |
| useState       | 状态管理             | [state, setState]                 |
| useEffect      | 副作用处理           | undefined                         |
| useContext     | 读取 Context         | context value                     |
| useReducer     | 复杂状态管理         | [state, dispatch]                 |
| useMemo        | 缓存计算结果         | memoized value                    |
| useCallback    | 缓存函数             | memoized function                 |
| useRef         | DOM 引用/保存值      | `{ current: value }`              |
| use            | 读取 Promise/Context | resolved value                    |
| useActionState | 管理 Action 状态     | [state, action, pending]          |
| useOptimistic  | 乐观更新             | [optimisticState, addOptimistic]  |
| useFormStatus  | 表单状态             | { pending, data, method, action } |

## 💡 最佳实践

### 1. 合理使用 useEffect 依赖

```jsx
// ✗ 缺少依赖
useEffect(() => {
  console.log(count);
}, []); // count 改变不会触发

// ✓ 正确的依赖
useEffect(() => {
  console.log(count);
}, [count]);
```

### 2. 避免不必要的渲染

```jsx
// ✗ 每次都创建新函数
function Parent() {
  return <Child onClick={() => console.log("click")} />;
}

// ✓ 使用 useCallback
function Parent() {
  const handleClick = useCallback(() => {
    console.log("click");
  }, []);

  return <Child onClick={handleClick} />;
}
```

### 3. 自定义 Hook 命名规范

```jsx
// ✓ 以 use 开头
function useWindowSize() {}
function useAuth() {}

// ✗ 不以 use 开头
function getWindowSize() {} // 不是 Hook
```

---

**下一步**: 查看 [React 19 新特性](/docs/react/react19-features) 了解最新的 Hook 功能，或浏览 [面试题精选](/docs/interview/react-interview-questions) 巩固知识。
