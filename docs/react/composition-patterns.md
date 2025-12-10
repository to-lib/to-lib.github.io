---
sidebar_position: 18
title: 组件组合模式
---

# 组件组合模式

> [!TIP]
> 组件组合是 React 中实现代码复用的强大方式。掌握这些模式能帮助你构建更灵活、可维护的组件。

## 📚 组合 vs 继承

React 推荐使用组合而非继承来复用组件逻辑。

```jsx
// ✗ 不推荐：使用继承
class BaseButton extends React.Component {}
class PrimaryButton extends BaseButton {}

// ✓ 推荐：使用组合
function Button({ variant, children }) {
  return <button className={`btn btn-${variant}`}>{children}</button>;
}
```

## 🎯 包含关系（Containment）

某些组件无法提前知道它们的子组件内容。

### children 属性

```jsx
function Card({ children }) {
  return (
    <div className="card">
      <div className="card-body">{children}</div>
    </div>
  );
}

// 使用
function App() {
  return (
    <Card>
      <h2>标题</h2>
      <p>这是卡片内容</p>
    </Card>
  );
}
```

### 多个"插槽"

```jsx
function SplitPane({ left, right }) {
  return (
    <div className="split-pane">
      <div className="split-pane-left">{left}</div>
      <div className="split-pane-right">{right}</div>
    </div>
  );
}

// 使用
function App() {
  return <SplitPane left={<Sidebar />} right={<MainContent />} />;
}
```

## 🔧 特例关系（Specialization）

有时组件是其他组件的"特殊实例"。

```jsx
// 通用对话框
function Dialog({ title, message, children }) {
  return (
    <div className="dialog">
      <h2>{title}</h2>
      <p>{message}</p>
      {children}
    </div>
  );
}

// 欢迎对话框（特例）
function WelcomeDialog() {
  return (
    <Dialog title="欢迎" message="感谢访问我们的应用！">
      <button>开始</button>
    </Dialog>
  );
}
```

## 🎨 复合组件模式（Compound Components）

让多个组件协同工作，共享状态。

### 基础实现

```jsx
import { createContext, useContext, useState } from "react";

// 创建上下文
const TabsContext = createContext();

// 主组件
function Tabs({ children, defaultValue }) {
  const [activeTab, setActiveTab] = useState(defaultValue);

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className="tabs">{children}</div>
    </TabsContext.Provider>
  );
}

// 子组件
Tabs.List = function TabsList({ children }) {
  return <div className="tabs-list">{children}</div>;
};

Tabs.Trigger = function TabsTrigger({ value, children }) {
  const { activeTab, setActiveTab } = useContext(TabsContext);

  return (
    <button
      className={activeTab === value ? "active" : ""}
      onClick={() => setActiveTab(value)}
    >
      {children}
    </button>
  );
};

Tabs.Content = function TabsContent({ value, children }) {
  const { activeTab } = useContext(TabsContext);

  if (value !== activeTab) return null;

  return <div className="tabs-content">{children}</div>;
};

// 使用
function App() {
  return (
    <Tabs defaultValue="tab1">
      <Tabs.List>
        <Tabs.Trigger value="tab1">标签1</Tabs.Trigger>
        <Tabs.Trigger value="tab2">标签2</Tabs.Trigger>
        <Tabs.Trigger value="tab3">标签3</Tabs.Trigger>
      </Tabs.List>

      <Tabs.Content value="tab1">标签1的内容</Tabs.Content>
      <Tabs.Content value="tab2">标签2的内容</Tabs.Content>
      <Tabs.Content value="tab3">标签3的内容</Tabs.Content>
    </Tabs>
  );
}
```

### 实战示例：下拉菜单

```jsx
const DropdownContext = createContext();

function Dropdown({ children }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <DropdownContext.Provider value={{ isOpen, setIsOpen }}>
      <div className="dropdown">{children}</div>
    </DropdownContext.Provider>
  );
}

Dropdown.Trigger = function DropdownTrigger({ children }) {
  const { isOpen, setIsOpen } = useContext(DropdownContext);

  return <button onClick={() => setIsOpen(!isOpen)}>{children}</button>;
};

Dropdown.Menu = function DropdownMenu({ children }) {
  const { isOpen } = useContext(DropdownContext);

  if (!isOpen) return null;

  return <div className="dropdown-menu">{children}</div>;
};

Dropdown.Item = function DropdownItem({ onClick, children }) {
  const { setIsOpen } = useContext(DropdownContext);

  return (
    <button
      className="dropdown-item"
      onClick={() => {
        onClick?.();
        setIsOpen(false);
      }}
    >
      {children}
    </button>
  );
};

// 使用
<Dropdown>
  <Dropdown.Trigger>菜单</Dropdown.Trigger>
  <Dropdown.Menu>
    <Dropdown.Item onClick={() => console.log("编辑")}>编辑</Dropdown.Item>
    <Dropdown.Item onClick={() => console.log("删除")}>删除</Dropdown.Item>
  </Dropdown.Menu>
</Dropdown>;
```

## 🎭 Render Props 模式

通过 props 传递渲染函数。

### 基础用法

```jsx
function Mouse({ render }) {
  const [position, setPosition] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e) => {
    setPosition({ x: e.clientX, y: e.clientY });
  };

  return <div onMouseMove={handleMouseMove}>{render(position)}</div>;
}

// 使用
<Mouse
  render={({ x, y }) => (
    <h1>
      鼠标位置：{x}, {y}
    </h1>
  )}
/>;
```

### children 作为函数

```jsx
function DataProvider({ children, url }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        setLoading(false);
      });
  }, [url]);

  return children({ data, loading });
}

// 使用
<DataProvider url="/api/users">
  {({ data, loading }) =>
    loading ? <div>加载中...</div> : <UserList users={data} />
  }
</DataProvider>;
```

## 🔄 高阶组件（HOC）

高阶组件是参数为组件，返回值为新组件的函数。

### 基础 HOC

```jsx
// 高阶组件：添加加载状态
function withLoading(Component) {
  return function WithLoadingComponent({ isLoading, ...props }) {
    if (isLoading) {
      return <div>加载中...</div>;
    }
    return <Component {...props} />;
  };
}

// 使用
const UserListWithLoading = withLoading(UserList);

function App() {
  const [loading, setLoading] = useState(true);
  const [users, setUsers] = useState([]);

  return <UserListWithLoading isLoading={loading} users={users} />;
}
```

### 实用 HOC 示例

```jsx
// 权限控制 HOC
function withAuth(Component) {
  return function WithAuthComponent(props) {
    const { user } = useAuth(); // 假设有这个 hook

    if (!user) {
      return <Navigate to="/login" />;
    }

    return <Component {...props} />;
  };
}

// 主题 HOC
function withTheme(Component) {
  return function WithThemeComponent(props) {
    const theme = useContext(ThemeContext);
    return <Component {...props} theme={theme} />;
  };
}

// 组合多个 HOC
const EnhancedComponent = withAuth(withTheme(MyComponent));
```

## 🎁 自定义 Hooks（推荐）

现代 React 推荐使用自定义 Hooks 代替 HOC。

### Hook vs HOC

```jsx
// ✗ HOC 方式
function withWindowSize(Component) {
  return function WithWindowSizeComponent(props) {
    const [size, setSize] = useState({
      width: window.innerWidth,
      height: window.innerHeight,
    });

    useEffect(() => {
      const handleResize = () => {
        setSize({
          width: window.innerWidth,
          height: window.innerHeight,
        });
      };
      window.addEventListener("resize", handleResize);
      return () => window.removeEventListener("resize", handleResize);
    }, []);

    return <Component {...props} windowSize={size} />;
  };
}

// ✓ Hook 方式（推荐）
function useWindowSize() {
  const [size, setSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  useEffect(() => {
    const handleResize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return size;
}

// 使用更简洁
function MyComponent() {
  const { width, height } = useWindowSize();
  return (
    <div>
      窗口: {width} x {height}
    </div>
  );
}
```

## 💡 最佳实践

### 1. 选择合适的模式

| 模式             | 适用场景             | 优点           | 缺点         |
| ---------------- | -------------------- | -------------- | ------------ |
| **组合**         | 简单的父子关系       | 简单直观       | -            |
| **Compound**     | 相关组件协同         | 灵活、API 简洁 | 需要 Context |
| **Render Props** | 需要动态渲染         | 灵活           | 嵌套过深     |
| **HOC**          | 横切关注点（旧项目） | 可组合         | Props 冲突   |
| **Hooks**        | 逻辑复用（推荐）     | 简洁、组合性好 | -            |

### 2. 避免过度嵌套

```jsx
// ✗ 不好：Render Props 地狱
<DataProvider>
  {(data) => (
    <ThemeProvider>
      {(theme) => (
        <AuthProvider>
          {(auth) => <Component data={data} theme={theme} auth={auth} />}
        </AuthProvider>
      )}
    </ThemeProvider>
  )}
</DataProvider>;

// ✓ 好：使用 Hooks
function Component() {
  const data = useData();
  const theme = useTheme();
  const auth = useAuth();

  return <div>...</div>;
}
```

### 3. Props 透传

```jsx
// ✓ 使用扩展运算符透传 props
function Button({ variant, ...props }) {
  return <button className={`btn btn-${variant}`} {...props} />;
}
```

---

**下一步**：学习 [React Router](/docs/react/react-router) 管理应用路由，或查看 [状态管理](/docs/react/state-management) 了解全局状态方案。
