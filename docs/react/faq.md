---
sidebar_position: 92
title: FAQ
---

# React 常见问题

> [!TIP]
> 本文汇总了 React 开发中最常见的问题和解答。

## 🔧 开发环境

### Q: npm start 失败怎么办？

```bash
# 清除依赖重新安装
rm -rf node_modules package-lock.json
npm install

# 清除 npm 缓存
npm cache clean --force
```

### Q: 端口被占用？

```bash
# macOS/Linux
lsof -ti:3000 | xargs kill -9

# Windows
netstat -ano | findstr :3000
taskkill /PID [进程ID] /F
```

## ⚛️ React 核心

### Q: 何时使用 useState vs useReducer?

- **useState**: 简单的独立状态
- **useReducer**: 复杂的相关状态、多个子值、复杂的状态逻辑

```jsx
// 简单情况：useState
const [count, setCount] = useState(0);

// 复杂情况：useReducer
const [state, dispatch] = useReducer(reducer, {
  count: 0,
  step: 1,
  history: [],
});
```

### Q: 为什么我的状态没有更新？

```jsx
// ✗ 错误：直接修改状态
const [items, setItems] = useState([1, 2, 3]);
items.push(4); // 不会触发重新渲染！

// ✓ 正确：创建新对象/数组
setItems([...items, 4]);
```

### Q: useEffect 为什么执行两次？

React 18 的严格模式会故意让组件挂载两次（仅开发环境），用于发现副作用问题。

```jsx
// 生产环境只执行一次
useEffect(() => {
  console.log("mounted"); // 开发环境打印两次
}, []);
```

### Q: 如何在 useEffect 中使用异步函数？

```jsx
// ✓ 方法1：内部定义 async 函数
useEffect(() => {
  async function fetchData() {
    const data = await fetch("/api");
    setData(data);
  }
  fetchData();
}, []);

// ✓ 方法2：立即执行的 async 函数
useEffect(() => {
  (async () => {
    const data = await fetch("/api");
    setData(data);
  })();
}, []);

// ✗ 错误：useEffect 回调不能是 async
useEffect(async () => {
  const data = await fetch("/api"); // 错误！
}, []);
```

## 🎨 组件与 Props

### Q: props 什么时候会改变？

props 是只读的，由父组件控制：

```jsx
// 父组件更新 props
function Parent() {
  const [count, setCount] = useState(0);
  return <Child count={count} />; // count 变化时，Child 会重新渲染
}

function Child({ count }) {
  // 不能修改 props
  // count = 123; // 错误！
  return <div>{count}</div>;
}
```

### Q: 如何传递大量 Props？

```jsx
// ✓ 使用扩展运算符
const props = { name: 'John', age: 30, email: 'john@example.com' };
<Component {...props} />

// ✓ 或使用对象
<Component user={{ name: 'John', age: 30, email: '...' }} />
```

## ⚡ 性能

### Q: 如何避免不必要的重新渲染？

```jsx
// 1. React.memo
const Child = React.memo(({ data }) => {
  return <div>{data}</div>;
});

// 2. useMemo 缓存计算
const expensiveResult = useMemo(() => {
  return computeExpensiveValue(a, b);
}, [a, b]);

// 3. useCallback 缓存函数
const handleClick = useCallback(() => {
  doSomething(a, b);
}, [a, b]);
```

### Q: 列表为什么需要 key？

key 帮助 React 识别哪些元素改变、添加或移除：

```jsx
// ✓ 好：使用稳定的 ID
{
  items.map((item) => <div key={item.id}>{item.name}</div>);
}

// ✗ 不好：使用索引（当列表会重新排序时）
{
  items.map((item, index) => <div key={index}>{item.name}</div>);
}
```

## 🔄 状态管理

### Q: Context 什么时候会重新渲染？

Context Provider 的 value 改变时，所有消费者都会重新渲染：

```jsx
// ✗ 不好：每次 Parent 渲染都创建新对象
function Parent() {
  const [user, setUser] = useState(null);
  return (
    <UserContext.Provider value={{ user, setUser }}>
      {children}
    </UserContext.Provider>
  );
}

// ✓ 好：使用 useMemo
const value = useMemo(() => ({ user, setUser }), [user]);
<UserContext.Provider value={value}>
```

### Q: 何时使用全局状态管理？

- ✓ 需要在多个不相关组件间共享状态
- ✓ 状态需要持久化
- ✓ 需要复杂的状态更新逻辑
- ✗ 简单的父子通信（用 props）
- ✗ 只在一个组件使用的状态

## 🧪 调试

### Q: 如何调试组件？

```jsx
// 1. 使用 React DevTools
// 安装浏览器扩展

// 2. console.log
function Component({ prop }) {
  console.log("Component rendered", { prop });
  return <div>{prop}</div>;
}

// 3. debugger 语句
useEffect(() => {
  debugger; // 断点
  // ...
}, []);
```

### Q: Warning: Can't perform a React state update on an unmounted component

组件卸载后尝试更新状态：

```jsx
// ✓ 解决：使用清理标志
useEffect(() => {
  let cancelled = false;

  fetch("/api").then((data) => {
    if (!cancelled) {
      setData(data);
    }
  });

  return () => {
    cancelled = true;
  };
}, []);
```

## 📚 学习资源

- [官方文档](https://react.dev)
- [React DevTools](https://react.dev/learn/react-developer-tools)
- [本站其他文档](/docs/react)

---

**还有问题？** 查看 [最佳实践](/docs/react/best-practices) 或 [面试题](/docs/react/interview-questions)
