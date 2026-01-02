---
sidebar_position: 19
title: 严格模式
---

# React 严格模式 (StrictMode)

> [!TIP]
> StrictMode 是一个用于突出显示应用程序中潜在问题的工具。它不会渲染任何可见的 UI，只是为其后代元素触发额外的检查和警告。

## 🎯 什么是 StrictMode？

StrictMode 是一个开发辅助工具，帮助你：

- 识别不安全的生命周期方法
- 检测意外的副作用
- 发现废弃的 API 使用
- 确保组件符合并发渲染要求

```jsx
import { StrictMode } from "react";

function App() {
  return (
    <StrictMode>
      <MyApp />
    </StrictMode>
  );
}
```

> [!IMPORTANT]
> StrictMode **仅在开发模式**下运行，不会影响生产构建。

## 🔄 双重渲染检测

StrictMode 会故意**双重调用**以下函数来检测副作用：

- 组件函数体
- useState、useMemo、useReducer 的初始化函数
- 类组件的 constructor、render、shouldComponentUpdate 等

### 为什么双重渲染？

```jsx
function Counter() {
  const [count, setCount] = useState(0);

  // ❌ 问题：每次渲染都会执行副作用
  console.log("Rendered!"); // StrictMode 下会打印两次

  // ✅ 正确：副作用应该在 useEffect 中
  useEffect(() => {
    console.log("Effect ran!");
  }, []);

  return <button onClick={() => setCount(count + 1)}>{count}</button>;
}
```

### 常见问题场景

```jsx
// ❌ 问题：渲染期间修改外部变量
let externalCount = 0;

function BadComponent() {
  externalCount++; // 双重渲染会导致计数不准确
  return <div>{externalCount}</div>;
}

// ✅ 正确：使用 state 管理
function GoodComponent() {
  const [count, setCount] = useState(0);
  return <div>{count}</div>;
}
```

## 🔁 Effect 双重执行

React 18+ 的 StrictMode 会**模拟组件卸载再重新挂载**，这意味着：

1. 组件挂载
2. Effect 执行
3. 清理函数执行（模拟卸载）
4. Effect 再次执行（模拟重新挂载）

### 为什么这样做？

这帮助你发现 Effect 清理不当的问题：

```jsx
// ❌ 问题：没有清理订阅
function ChatRoom({ roomId }) {
  useEffect(() => {
    const connection = createConnection(roomId);
    connection.connect();
    // 缺少清理函数！
  }, [roomId]);
}

// ✅ 正确：正确清理
function ChatRoom({ roomId }) {
  useEffect(() => {
    const connection = createConnection(roomId);
    connection.connect();

    return () => {
      connection.disconnect(); // 清理连接
    };
  }, [roomId]);
}
```

### 处理双重执行

```jsx
function DataFetcher({ url }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    let cancelled = false; // 取消标志

    fetch(url)
      .then((res) => res.json())
      .then((result) => {
        if (!cancelled) {
          // 检查是否已取消
          setData(result);
        }
      });

    return () => {
      cancelled = true; // 取消请求
    };
  }, [url]);

  return <div>{data ? JSON.stringify(data) : "Loading..."}</div>;
}
```

## ⚠️ 废弃 API 警告

StrictMode 会警告使用已废弃的 API：

### React 19 中已移除的 API

| 废弃 API           | 替代方案                     |
| ------------------ | ---------------------------- |
| `findDOMNode`      | 使用 `ref`                   |
| 字符串 refs        | 使用 `useRef` 或 `createRef` |
| Legacy Context     | 使用 `createContext`         |
| `UNSAFE_` 生命周期 | 使用 Hooks                   |

```jsx
// ❌ 废弃：字符串 ref
class OldComponent extends React.Component {
  componentDidMount() {
    this.refs.myInput.focus(); // 废弃
  }

  render() {
    return <input ref="myInput" />;
  }
}

// ✅ 推荐：使用 useRef
function NewComponent() {
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current.focus();
  }, []);

  return <input ref={inputRef} />;
}
```

## 📦 局部使用

你可以只对部分组件树启用 StrictMode：

```jsx
function App() {
  return (
    <div>
      <Header /> {/* 不受 StrictMode 影响 */}
      <StrictMode>
        <main>
          <NewFeature /> {/* 受 StrictMode 检查 */}
        </main>
      </StrictMode>
      <Footer /> {/* 不受 StrictMode 影响 */}
    </div>
  );
}
```

## 🛠️ 常见问题解答

### 为什么我的组件渲染了两次？

这是 StrictMode 的正常行为，用于检测副作用问题。在生产环境中只会渲染一次。

### 为什么我的 useEffect 执行了两次？

StrictMode 模拟卸载再挂载，确保你的 Effect 正确清理。这帮助发现内存泄漏等问题。

### 如何关闭双重渲染？

不建议关闭，因为它帮助发现潜在问题。如果某个组件确实有问题，应该修复组件而不是关闭 StrictMode。

### 生产环境有影响吗？

没有。StrictMode 的所有检查只在开发模式下运行。

## 💡 最佳实践

### 1. 在根组件启用

```jsx
// main.jsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

### 2. 确保 Effect 可重复执行

```jsx
function Timer() {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setSeconds((s) => s + 1);
    }, 1000);

    return () => clearInterval(id); // 必须清理
  }, []);

  return <div>{seconds}</div>;
}
```

### 3. 避免渲染期间的副作用

```jsx
// ❌ 避免
function Bad() {
  localStorage.setItem("rendered", "true"); // 渲染期间产生副作用
  return <div>...</div>;
}

// ✅ 推荐
function Good() {
  useEffect(() => {
    localStorage.setItem("rendered", "true"); // 在 Effect 中
  }, []);
  return <div>...</div>;
}
```

## 🔗 相关资源

- [并发渲染](/docs/react/concurrent-rendering)
- [Hooks 详解](/docs/react/hooks)
- [React 19 新特性](/docs/react/react19-features)

---

**下一步**：了解 [React DevTools](/docs/react/devtools) 进行组件调试和性能分析。
