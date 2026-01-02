---
sidebar_position: 20
title: React DevTools
---

# React DevTools 使用指南

> [!TIP]
> React DevTools 是调试 React 应用的必备工具，可以检查组件树、分析性能、调试 Hooks 状态。

## 📦 安装

### 浏览器扩展（推荐）

- [Chrome 扩展](https://chrome.google.com/webstore/detail/react-developer-tools/)
- [Firefox 扩展](https://addons.mozilla.org/firefox/addon/react-devtools/)
- [Edge 扩展](https://microsoftedge.microsoft.com/addons/detail/react-developer-tools/)

### 独立版本

用于调试 React Native 或其他环境：

```bash
npm install -g react-devtools
react-devtools  # 启动独立窗口
```

## 🔍 Components 面板

### 组件树检查

Components 面板显示 React 组件树结构：

```
App
├── Header
│   ├── Logo
│   └── Navigation
├── Main
│   ├── Sidebar
│   └── Content
│       ├── ArticleList
│       │   ├── Article
│       │   └── Article
│       └── Pagination
└── Footer
```

### 查看组件信息

选中组件后可以查看：

| 选项卡          | 内容                          |
| --------------- | ----------------------------- |
| **props**       | 传入的属性                    |
| **hooks**       | useState、useEffect 等状态    |
| **rendered by** | 渲染该组件的父组件            |
| **source**      | 源代码位置（需要 source map） |

### 编辑 Props 和 State

```jsx
// 在 DevTools 中可以直接修改 state 值进行调试
function Counter() {
  const [count, setCount] = useState(0);
  // 在 DevTools 中可以修改 count 的值
  return <div>{count}</div>;
}
```

### 搜索组件

- 按名称搜索：直接输入组件名
- 按正则搜索：`/Article/`
- 按属性搜索：`props.id=123`

## ⚡ Profiler 面板

Profiler 用于分析组件渲染性能。

### 开始录制

1. 打开 Profiler 面板
2. 点击录制按钮（圆点）
3. 在应用中执行操作
4. 停止录制

### 分析结果

#### Flamegraph（火焰图）

显示组件渲染时间的层级视图：

- **宽度**：渲染时间
- **颜色**：相对耗时（灰色 = 未渲染，黄色 = 较慢，绿色 = 较快）

#### Ranked Chart

按渲染时间排序显示组件：

```
ArticleList       12.3ms
Article (x10)     8.5ms
Sidebar           2.1ms
Header            0.5ms
```

### 识别问题

```jsx
// ❌ 问题：每次父组件渲染都触发重渲染
function Parent() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <button onClick={() => setCount(count + 1)}>{count}</button>
      <ExpensiveChild /> {/* 每次都重新渲染 */}
    </div>
  );
}

// ✅ 优化：使用 memo
const ExpensiveChild = memo(function ExpensiveChild() {
  // 昂贵的渲染逻辑
  return <div>...</div>;
});
```

### 查看为什么重渲染

在 Profiler 设置中启用 **"Record why each component rendered while profiling"**：

- Props changed
- State changed
- Hooks changed
- Parent rendered

## 🎯 调试技巧

### 1. 高亮更新

在设置中启用 **"Highlight updates when components render"**：

- 蓝色边框 = 组件已更新
- 帮助识别不必要的重渲染

### 2. 隐藏原生 DOM 元素

过滤掉 `div`、`span` 等原生元素，只显示 React 组件。

### 3. 组件过滤

```jsx
// 在 DevTools 设置中添加过滤规则
// 隐藏特定组件：
// name: /^Styled/ (隐藏 styled-components)
// name: /^Provider$/ (隐藏 Context Providers)
```

### 4. 使用 $r 访问选中组件

在浏览器控制台中：

```js
// 选中组件后
$r; // 返回选中组件的 Fiber 节点
$r.memoizedState; // 查看 state
$r.memoizedProps; // 查看 props
```

## 🔧 高级功能

### Timeline（时间线）

在 Profiler 的 Timeline 视图中查看：

- 渲染开始和结束时间
- Suspense 边界状态
- 并发渲染中的优先级切换

### Suspense 调试

```jsx
function App() {
  return (
    <Suspense fallback={<Loading />}>
      <LazyComponent />
    </Suspense>
  );
}
// DevTools 会显示 Suspense 边界和 fallback 状态
```

### Server Components 标记

React 19 的 Server Components 在 DevTools 中会有特殊标记：

- 🖥️ Server Component
- 💻 Client Component

## 🐛 常见调试场景

### 1. 找出性能问题

```jsx
// 1. 打开 Profiler
// 2. 录制一段操作
// 3. 查看 Ranked Chart 找出最慢的组件
// 4. 优化该组件
```

### 2. 调试 Context 问题

```jsx
// DevTools 会显示 Context.Provider 和消费的值
// 可以直接查看和修改 Context 值
```

### 3. 检查 Hook 状态

```jsx
function MyComponent() {
  const [state1, setState1] = useState("a");
  const [state2, setState2] = useState("b");
  const ref = useRef(null);

  // DevTools 中显示：
  // hooks:
  //   State: "a"
  //   State: "b"
  //   Ref: { current: null }
}
```

### 4. 定位组件源码

1. 选中组件
2. 点击组件名右侧的 `<>` 图标
3. 自动跳转到源代码位置

## 💡 最佳实践

### 使用 displayName

```jsx
// 为匿名组件添加名称
const MyComponent = memo(function MyComponent() {
  return <div>...</div>;
});

// 或使用 displayName
const MyComponent = memo(() => <div>...</div>);
MyComponent.displayName = "MyComponent";
```

### 环境检测

```jsx
// 仅在开发环境启用额外调试信息
if (process.env.NODE_ENV === "development") {
  console.log("Debug info:", someData);
}
```

### 配合 React.Profiler 组件

```jsx
function onRenderCallback(
  id, // Profiler 标识
  phase, // "mount" 或 "update"
  actualDuration, // 本次渲染耗时
  baseDuration, // 无 memo 时的预估耗时
  startTime,
  commitTime
) {
  console.log(`${id} ${phase}: ${actualDuration}ms`);
}

function App() {
  return (
    <Profiler id="App" onRender={onRenderCallback}>
      <MyApp />
    </Profiler>
  );
}
```

## 🔗 相关资源

- [严格模式](/docs/react/strict-mode)
- [性能优化](/docs/react/performance-optimization)
- [测试](/docs/react/testing)

---

**下一步**：学习 [性能优化](/docs/react/performance-optimization) 技巧，配合 DevTools 提升应用性能。
