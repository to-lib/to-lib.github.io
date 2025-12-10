---
sidebar_position: 11
title: 性能优化
---

# React 性能优化

> [!TIP]
> 性能优化是构建高性能 React 应用的关键，本文介绍常用的优化技巧和最佳实践。

## 📊 性能分析

### React DevTools Profiler

```jsx
// 使用 Profiler 测量性能
import { Profiler } from "react";

function App() {
  const onRenderCallback = (
    id,
    phase,
    actualDuration,
    baseDuration,
    startTime,
    commitTime,
    interactions
  ) => {
    console.log(`${id} took ${actualDuration}ms`);
  };

  return (
    <Profiler id="App" onRender={onRenderCallback}>
      <Navigation />
      <Main />
    </Profiler>
  );
}
```

## 🎯 React.memo

### 基础用法

```jsx
// ✗ 每次父组件渲染都会重新渲染
function Child({ name }) {
  console.log("Child rendered");
  return <div>{name}</div>;
}

// ✓ 使用 React.memo 避免不必要的渲染
const Child = React.memo(function Child({ name }) {
  console.log("Child rendered");
  return <div>{name}</div>;
});

function Parent() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <button onClick={() => setCount(count + 1)}>Count: {count}</button>
      <Child name="Alice" /> {/* name 不变，不会重新渲染 */}
    </div>
  );
}
```

### 自定义比较函数

```jsx
const Child = React.memo(
  function Child({ user }) {
    return <div>{user.name}</div>;
  },
  (prevProps, nextProps) => {
    // 返回 true 表示不需要重新渲染
    return prevProps.user.id === nextProps.user.id;
  }
);
```

## ⚡ useMemo

### 缓存计算结果

```jsx
function TodoList({ todos, filter }) {
  // ✗ 每次渲染都会过滤
  const filteredTodos = todos.filter((todo) =>
    filter === "active" ? !todo.done : todo.done
  );

  // ✓ 使用 useMemo 缓存结果
  const filteredTodos = useMemo(() => {
    console.log("Filtering todos...");
    return todos.filter((todo) =>
      filter === "active" ? !todo.done : todo.done
    );
  }, [todos, filter]);

  return (
    <ul>
      {filteredTodos.map((todo) => (
        <li key={todo.id}>{todo.text}</li>
      ))}
    </ul>
  );
}
```

### 缓存复杂对象

```jsx
function Map({ markers }) {
  // ✗ 每次渲染都创建新对象
  const bounds = {
    ne: { lat: 10, lng: 10 },
    sw: { lat: 0, lng: 0 },
  };

  // ✓ 使用 useMemo
  const bounds = useMemo(
    () => ({
      ne: { lat: 10, lng: 10 },
      sw: { lat: 0, lng: 0 },
    }),
    []
  );

  return <MapComponent bounds={bounds} />;
}
```

## 🔄 useCallback

### 缓存回调函数

```jsx
function Parent() {
  const [count, setCount] = useState(0);

  // ✗ 每次渲染都创建新函数
  const handleClick = () => {
    console.log("Clicked");
  };

  // ✓ 使用 useCallback
  const handleClick = useCallback(() => {
    console.log("Clicked");
  }, []);

  return (
    <div>
      <button onClick={() => setCount(count + 1)}>Count: {count}</button>
      <Child onClick={handleClick} />
    </div>
  );
}

const Child = React.memo(({ onClick }) => {
  console.log("Child rendered");
  return <button onClick={onClick}>Click Me</button>;
});
```

### 带依赖的回调

```jsx
function SearchBox() {
  const [query, setQuery] = useState("");

  const handleSearch = useCallback(() => {
    console.log("Searching for:", query);
    // 执行搜索
  }, [query]); // query 变化时更新函数

  return (
    <div>
      <input value={query} onChange={(e) => setQuery(e.target.value)} />
      <SearchButton onSearch={handleSearch} />
    </div>
  );
}
```

## 🧩 代码分割

### React.lazy 和 Suspense

```jsx
import { lazy, Suspense } from "react";

// 懒加载组件
const HeavyComponent = lazy(() => import("./HeavyComponent"));

function App() {
  return (
    <div>
      <Suspense fallback={<div>Loading...</div>}>
        <HeavyComponent />
      </Suspense>
    </div>
  );
}
```

### 路由级代码分割

```jsx
import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

const Home = lazy(() => import("./pages/Home"));
const About = lazy(() => import("./pages/About"));
const Contact = lazy(() => import("./pages/Contact"));

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<LoadingSpinner />}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
```

## 📋 虚拟化长列表

### react-window

```jsx
import { FixedSizeList } from "react-window";

function VirtualList({ items }) {
  const Row = ({ index, style }) => (
    <div style={style} className="list-item">
      {items[index].name}
    </div>
  );

  return (
    <FixedSizeList
      height={400}
      itemCount={items.length}
      itemSize={35}
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
}
```

### react-virtualized

```jsx
import { List } from "react-virtualized";

function VirtualizedList({ items }) {
  const rowRenderer = ({ key, index, style }) => (
    <div key={key} style={style}>
      {items[index].name}
    </div>
  );

  return (
    <List
      width={300}
      height={400}
      rowCount={items.length}
      rowHeight={35}
      rowRenderer={rowRenderer}
    />
  );
}
```

## 🎨 避免不必要的渲染

### 1. 避免内联对象和数组

```jsx
// ✗ 不好 - 每次渲染创建新对象
function Component() {
  return <Child style={{ color: "red" }} items={["a", "b"]} />;
}

// ✓ 好 - 提取到外部
const style = { color: "red" };
const items = ["a", "b"];

function Component() {
  return <Child style={style} items={items} />;
}
```

### 2. 使用 key 优化列表

```jsx
// ✓ 使用稳定的 key
{
  items.map((item) => <Item key={item.id} data={item} />);
}

// ✗ 避免使用 index
{
  items.map((item, index) => <Item key={index} data={item} />);
}
```

### 3. 状态下放

```jsx
// ✗ 不好 - 父组件管理所有状态
function Parent() {
  const [input1, setInput1] = useState("");
  const [input2, setInput2] = useState("");

  return (
    <div>
      <ExpensiveComponent /> {/* 每次输入都重新渲染 */}
      <input value={input1} onChange={(e) => setInput1(e.target.value)} />
      <input value={input2} onChange={(e) => setInput2(e.target.value)} />
    </div>
  );
}

// ✓ 好 - 状态下放到子组件
function Parent() {
  return (
    <div>
      <ExpensiveComponent /> {/* 不会重新渲染 */}
      <InputForm />
    </div>
  );
}

function InputForm() {
  const [input1, setInput1] = useState("");
  const [input2, setInput2] = useState("");

  return (
    <>
      <input value={input1} onChange={(e) => setInput1(e.target.value)} />
      <input value={input2} onChange={(e) => setInput2(e.target.value)} />
    </>
  );
}
```

## 🚀 其他优化技巧

### 1. 防抖和节流

```jsx
import { debounce } from "lodash";

function SearchBox() {
  const [results, setResults] = useState([]);

  const searchAPI = useCallback(
    debounce(async (query) => {
      const data = await fetch(`/api/search?q=${query}`);
      setResults(data);
    }, 300),
    []
  );

  return (
    <input
      onChange={(e) => searchAPI(e.target.value)}
      placeholder="Search..."
    />
  );
}
```

### 2. Web Workers

```jsx
function HeavyComputation() {
  const [result, setResult] = useState(null);

  useEffect(() => {
    const worker = new Worker("worker.js");

    worker.postMessage({ data: largeDataSet });

    worker.onmessage = (e) => {
      setResult(e.data);
    };

    return () => worker.terminate();
  }, []);

  return <div>{result}</div>;
}
```

### 3. 图片优化

```jsx
// 懒加载图片
function LazyImage({ src, alt }) {
  const [imageSrc, setImageSrc] = useState(null);
  const imgRef = useRef();

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setImageSrc(src);
        observer.disconnect();
      }
    });

    observer.observe(imgRef.current);

    return () => observer.disconnect();
  }, [src]);

  return <img ref={imgRef} src={imageSrc || "placeholder.jpg"} alt={alt} />;
}
```

## 💡 性能检查清单

### 渲染优化

- [ ] 使用 React.memo 包装纯组件
- [ ] 使用 useMemo 缓存昂贵计算
- [ ] 使用 useCallback 缓存回调函数
- [ ] 避免内联对象和函数
- [ ] 列表使用稳定的 key

### 代码优化

- [ ] 使用 React.lazy 代码分割
- [ ] 路由级懒加载
- [ ] 使用虚拟化长列表
- [ ] 防抖/节流高频操作

### 资源优化

- [ ] 图片懒加载
- [ ] 压缩打包体积
- [ ] 使用 CDN
- [ ] 开启 Gzip

## 📖 实用示例

### 优化大型表格

```jsx
const TableRow = React.memo(({ row, onEdit, onDelete }) => {
  return (
    <tr>
      <td>{row.id}</td>
      <td>{row.name}</td>
      <td>{row.email}</td>
      <td>
        <button onClick={() => onEdit(row.id)}>Edit</button>
        <button onClick={() => onDelete(row.id)}>Delete</button>
      </td>
    </tr>
  );
});

function Table({ data }) {
  const handleEdit = useCallback((id) => {
    console.log("Edit:", id);
  }, []);

  const handleDelete = useCallback((id) => {
    console.log("Delete:", id);
  }, []);

  return (
    <table>
      <tbody>
        {data.map((row) => (
          <TableRow
            key={row.id}
            row={row}
            onEdit={handleEdit}
            onDelete={handleDelete}
          />
        ))}
      </tbody>
    </table>
  );
}
```

---

**下一步**: 学习 [状态管理](/docs/react/state-management) 了解全局状态管理方案，或查看 [TypeScript](/docs/react/typescript) 提升代码质量。
