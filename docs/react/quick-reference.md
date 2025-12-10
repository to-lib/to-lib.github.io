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

---

**更多详情**：查看 [Hooks 详解](/docs/react/hooks) 或 [最佳实践](/docs/react/best-practices)
