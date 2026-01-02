---
sidebar_position: 90
title: 快速参考
---

# React 快速参考

> [!TIP]
> 本文档提供 React 常用 API、Hooks 和模式的快速查询。

## 🎯 核心 Hooks

### useState

```jsx
const [state, setState] = useState(initialValue);
setState(newValue); // 直接设置
setState((prev) => prev + 1); // 函数式更新
```

### useEffect

```jsx
useEffect(() => {  // 每次渲染
useEffect(() => {}, []); // 仅挂载
useEffect(() => {}, [dep]); // dep 变化
useEffect(() => { return () => {} }, []); // 清理函数
```

### useContext

```jsx
const value = useContext(MyContext);
```

### useRef

```jsx
const ref = useRef(initialValue);
ref.current = newValue; // 不触发重新渲染
<input ref={ref} />;
```

### useMemo / useCallback

```jsx
const memoizedValue = useMemo(() => computeExpensiveValue(a, b), [a, b]);
const memoizedCallback = useCallback(() => {
  doSomething(a, b);
}, [a, b]);
```

## 📝 组件模式

### 函数组件

```jsx
function Component({ prop1, prop2 }) {
  return <div>{prop1}</div>;
}
```

### Props 类型

```tsx
interface Props {
  required: string;
  optional?: number;
  children?: React.ReactNode;
  onClick?: () => void;
}
```

### 条件渲染

```jsx
{
  condition && <Component />;
}
{
  condition ? <A /> : <B />;
}
```

### 列表渲染

```jsx
{
  items.map((item) => <Item key={item.id} {...item} />);
}
```

## 🎨 常用事件

```tsx
onClick={(e: React.MouseEvent) => {}}
onChange={(e: React.ChangeEvent<HTMLInputElement>) => {}}
onSubmit={(e: React.FormEvent) => { e.preventDefault(); }}
```

## 🔧 实用代码片段

### 数据获取

```jsx
const [data, setData] = useState(null);
const [loading, setLoading] = useState(true);

useEffect(() => {
  fetch(url)
    .then((r) => r.json())
    .then((d) => {
      setData(d);
      setLoading(false);
    });
}, [url]);
```

### 表单处理

```jsx
const [form, setForm] = useState({ name: "", email: "" });
const handleChange = (e) => {
  setForm({ ...form, [e.target.name]: e.target.value });
};
```

### 防抖/节流

```jsx
import { useDebounce } from "use-debounce";
const [value] = useDebounce(searchTerm, 500);
```

## 🆕 React 19 新 Hooks

### useFormStatus

```jsx
import { useFormStatus } from "react-dom";

function SubmitButton() {
  const { pending, data, method, action } = useFormStatus();
  return <button disabled={pending}>{pending ? "提交中..." : "提交"}</button>;
}
```

### useActionState

```jsx
import { useActionState } from "react";

async function createUser(prevState, formData) {
  const name = formData.get("name");
  // 返回新状态
  return { success: true, message: `已创建用户 ${name}` };
}

function Form() {
  const [state, formAction, isPending] = useActionState(createUser, {
    message: "",
  });
  return (
    <form action={formAction}>
      <input name="name" />
      <button type="submit" disabled={isPending}>
        {isPending ? "创建中..." : "创建"}
      </button>
      {state.message && <p>{state.message}</p>}
    </form>
  );
}
```

### useOptimistic

```jsx
import { useOptimistic, useTransition } from "react";

function TodoList({ todos }) {
  const [optimisticTodos, addOptimisticTodo] = useOptimistic(
    todos,
    (state, newTodo) => [...state, { ...newTodo, sending: true }]
  );

  async function addTodo(formData) {
    const newTodo = { id: Date.now(), text: formData.get("text") };
    addOptimisticTodo(newTodo);
    await saveTodo(newTodo);
  }

  return (
    <form action={addTodo}>
      <input name="text" />
      <button>添加</button>
    </form>
  );
}
```

### use

```jsx
import { use, Suspense } from "react";

// 读取 Promise
function UserProfile({ userPromise }) {
  const user = use(userPromise);
  return <div>{user.name}</div>;
}

// 读取 Context
function ThemeButton() {
  const theme = use(ThemeContext);
  return <button className={theme}>Click</button>;
}
```

## 🎨 更多事件类型

```tsx
// 鼠标事件
onClick: (e: React.MouseEvent<HTMLButtonElement>) => void
onDoubleClick: (e: React.MouseEvent) => void
onMouseEnter: (e: React.MouseEvent) => void
onMouseLeave: (e: React.MouseEvent) => void
onContextMenu: (e: React.MouseEvent) => void

// 键盘事件
onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void
onKeyUp: (e: React.KeyboardEvent) => void
onKeyPress: (e: React.KeyboardEvent) => void  // 已废弃

// 表单事件
onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
onSubmit: (e: React.FormEvent<HTMLFormElement>) => void
onFocus: (e: React.FocusEvent<HTMLInputElement>) => void
onBlur: (e: React.FocusEvent) => void

// 拖拽事件
onDrag: (e: React.DragEvent) => void
onDrop: (e: React.DragEvent) => void
onDragOver: (e: React.DragEvent) => void

// 触摸事件
onTouchStart: (e: React.TouchEvent) => void
onTouchMove: (e: React.TouchEvent) => void
onTouchEnd: (e: React.TouchEvent) => void

// 滚动事件
onScroll: (e: React.UIEvent<HTMLDivElement>) => void
```

## 📋 常用类型声明

```tsx
// 基础组件 Props
interface BaseProps {
  className?: string;
  style?: React.CSSProperties;
  children?: React.ReactNode;
}

// 通用按钮 Props
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
}

// 输入框 Props
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

// 模态框 Props
interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
}

// API 响应类型
interface ApiResponse<T> {
  data: T;
  error?: string;
  loading: boolean;
}

// 分页类型
interface Pagination {
  page: number;
  pageSize: number;
  total: number;
}

// 表格列定义
interface Column<T> {
  key: keyof T;
  title: string;
  render?: (value: T[keyof T], record: T) => React.ReactNode;
}
```

## 🛡️ 错误边界模板

```tsx
import { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("错误捕获:", error, errorInfo);
    // 上报错误到监控服务
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="error-fallback">
            <h2>出错了</h2>
            <p>{this.state.error?.message}</p>
            <button onClick={() => this.setState({ hasError: false })}>
              重试
            </button>
          </div>
        )
      );
    }
    return this.props.children;
  }
}

// 使用
<ErrorBoundary fallback={<div>加载失败</div>}>
  <MyComponent />
</ErrorBoundary>;
```

## 🔄 常用自定义 Hooks

```jsx
// useLocalStorage
function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    const saved = localStorage.getItem(key);
    return saved ? JSON.parse(saved) : initialValue;
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  return [value, setValue];
}

// useDebounce
function useDebounce(value, delay = 500) {
  const [debouncedValue, setDebouncedValue] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debouncedValue;
}

// useToggle
function useToggle(initialValue = false) {
  const [value, setValue] = useState(initialValue);
  const toggle = useCallback(() => setValue((v) => !v), []);
  return [value, toggle];
}

// useFetch
function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(url)
      .then((res) => res.json())
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [url]);

  return { data, loading, error };
}
```

---

**更多详情**：查看 [Hooks 详解](/docs/react/hooks) 或 [最佳实践](/docs/react/best-practices)
