---
sidebar_position: 8
title: 条件渲染
---

# 条件渲染

> [!TIP]
> 条件渲染让你根据不同的条件显示不同的内容，React 提供了多种方式实现条件渲染。

## 🎯 基础方法

### if 语句

```jsx
function Greeting({ isLoggedIn }) {
  if (isLoggedIn) {
    return <h1>Welcome back!</h1>;
  }
  return <h1>Please sign in.</h1>;
}
```

### 元素变量

```jsx
function LoginButton({ isLoggedIn, onLogin, onLogout }) {
  let button;

  if (isLoggedIn) {
    button = <button onClick={onLogout}>Logout</button>;
  } else {
    button = <button onClick={onLogin}>Login</button>;
  }

  return <div>{button}</div>;
}
```

## ⚡ 常用方法

### 三元运算符

```jsx
function Greeting({ isLoggedIn }) {
  return (
    <div>{isLoggedIn ? <h1>Welcome back!</h1> : <h1>Please sign in.</h1>}</div>
  );
}

// 内联样式
function Badge({ count }) {
  return (
    <span className={count > 0 ? "badge-active" : "badge-inactive"}>
      {count}
    </span>
  );
}
```

### 逻辑与 &&

```jsx
function Inbox({ unreadCount }) {
  return (
    <div>
      <h1>Messages</h1>
      {unreadCount > 0 && <p>You have {unreadCount} unread messages.</p>}
    </div>
  );
}

// 多个条件
function UserProfile({ user, isAdmin }) {
  return (
    <div>
      <h2>{user.name}</h2>
      {user.email && <p>Email: {user.email}</p>}
      {user.phone && <p>Phone: {user.phone}</p>}
      {isAdmin && <span className="admin-badge">Admin</span>}
    </div>
  );
}
```

### 逻辑或 ||

```jsx
function UserName({ user }) {
  return <h1>{user.name || "Guest"}</h1>;
}

// 空值合并
function UserAge({ user }) {
  return <p>Age: {user.age ?? "Unknown"}</p>;
}
```

## 🔄 多条件渲染

### Switch 语句

```jsx
function StatusMessage({ status }) {
  let message;

  switch (status) {
    case "loading":
      message = <p>Loading...</p>;
      break;
    case "success":
      message = <p className="success">Success!</p>;
      break;
    case "error":
      message = <p className="error">Error occurred</p>;
      break;
    default:
      message = <p>Ready</p>;
  }

  return <div>{message}</div>;
}
```

### 对象映射

```jsx
function StatusMessage({ status }) {
  const messages = {
    loading: <p>Loading...</p>,
    success: <p className="success">Success!</p>,
    error: <p className="error">Error occurred</p>,
    idle: <p>Ready</p>,
  };

  return <div>{messages[status] || messages.idle}</div>;
}
```

### 枚举对象

```jsx
const STATUS = {
  LOADING: "loading",
  SUCCESS: "success",
  ERROR: "error",
};

function StatusMessage({ status }) {
  const renderMessage = () => {
    switch (status) {
      case STATUS.LOADING:
        return <LoadingSpinner />;
      case STATUS.SUCCESS:
        return <SuccessIcon />;
      case STATUS.ERROR:
        return <ErrorMessage />;
      default:
        return null;
    }
  };

  return <div>{renderMessage()}</div>;
}
```

## 🎨 渲染列表中的条件

### 过滤数组

```jsx
function TodoList({ todos, filter }) {
  const filteredTodos = todos.filter((todo) => {
    if (filter === "active") return !todo.completed;
    if (filter === "completed") return todo.completed;
    return true; // 'all'
  });

  return (
    <ul>
      {filteredTodos.map((todo) => (
        <li key={todo.id}>{todo.text}</li>
      ))}
    </ul>
  );
}
```

### 条件样式

```jsx
function TodoItem({ todo }) {
  return (
    <li
      className={todo.completed ? "completed" : "active"}
      style={{
        textDecoration: todo.completed ? "line-through" : "none",
        color: todo.urgent ? "red" : "black",
      }}
    >
      {todo.text}
    </li>
  );
}
```

## 🚫 阻止渲染

### 返回 null

```jsx
function WarningBanner({ warn }) {
  if (!warn) {
    return null; // 不渲染任何内容
  }

  return <div className="warning">Warning!</div>;
}
```

### 条件包装器

```jsx
function ConditionalWrapper({ condition, wrapper, children }) {
  return condition ? wrapper(children) : children;
}

// 使用
<ConditionalWrapper
  condition={isHighlighted}
  wrapper={(children) => <div className="highlight">{children}</div>}
>
  <p>Content</p>
</ConditionalWrapper>;
```

## 💡 高级模式

### 渲染函数

```jsx
function DataDisplay({ data, renderLoading, renderError, renderData }) {
  if (data.loading) return renderLoading();
  if (data.error) return renderError(data.error);
  return renderData(data.result);
}

// 使用
<DataDisplay
  data={userData}
  renderLoading={() => <Spinner />}
  renderError={(error) => <ErrorMessage error={error} />}
  renderData={(data) => <UserProfile user={data} />}
/>;
```

### 短路渲染

```jsx
function Component({ user }) {
  return (
    <div>
      {/* 只有 user 存在才渲染 */}
      {user && (
        <>
          <h2>{user.name}</h2>
          <p>{user.email}</p>
        </>
      )}

      {/* 嵌套条件 */}
      {user && user.isPremium && <span className="premium-badge">Premium</span>}
    </div>
  );
}
```

### 早期返回

```jsx
function UserProfile({ user }) {
  // 早期返回处理特殊情况
  if (!user) {
    return <div>No user found</div>;
  }

  if (user.isBlocked) {
    return <div>User is blocked</div>;
  }

  // 正常渲染
  return (
    <div>
      <h2>{user.name}</h2>
      <p>{user.email}</p>
    </div>
  );
}
```

## ⚠️ 常见陷阱

### 避免使用 0

```jsx
// ✗ 问题 - 会渲染 "0"
function Component({ count }) {
  return <div>{count && <p>Count: {count}</p>}</div>;
}

// ✓ 正确
function Component({ count }) {
  return (
    <div>
      {count > 0 && <p>Count: {count}</p>}
      {/* 或 */}
      {!!count && <p>Count: {count}</p>}
    </div>
  );
}
```

### 避免复杂嵌套

```jsx
// ✗ 不好 - 嵌套太深
{
  isLoggedIn ? isAdmin ? <AdminPanel /> : <UserPanel /> : <LoginForm />;
}

// ✓ 好 - 提取函数
function renderPanel() {
  if (!isLoggedIn) return <LoginForm />;
  if (isAdmin) return <AdminPanel />;
  return <UserPanel />;
}

{
  renderPanel();
}
```

## 📖 实用示例

### 权限控制

```jsx
function ProtectedContent({ user, requiredRole, children }) {
  if (!user) {
    return <LoginPrompt />;
  }

  if (!user.roles.includes(requiredRole)) {
    return <AccessDenied />;
  }

  return <>{children}</>;
}

// 使用
<ProtectedContent user={currentUser} requiredRole="admin">
  <AdminDashboard />
</ProtectedContent>;
```

### 加载状态

```jsx
function DataFetcher({ url }) {
  const [state, setState] = useState({
    loading: true,
    error: null,
    data: null,
  });

  // ... fetch logic

  if (state.loading) {
    return <LoadingSpinner />;
  }

  if (state.error) {
    return <ErrorMessage error={state.error} />;
  }

  return <DataDisplay data={state.data} />;
}
```

### 空状态

```jsx
function ProductList({ products }) {
  if (products.length === 0) {
    return (
      <div className="empty-state">
        <img src="/empty-box.svg" alt="No products" />
        <h2>No products found</h2>
        <p>Try adjusting your filters</p>
        <button>Clear Filters</button>
      </div>
    );
  }

  return (
    <div className="product-grid">
      {products.map((product) => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  );
}
```

### 响应式渲染

```jsx
function ResponsiveMenu() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return <nav>{isMobile ? <MobileMenu /> : <DesktopMenu />}</nav>;
}
```

### 功能开关

```jsx
const FEATURES = {
  newDashboard: true,
  betaFeature: false,
  experimentalUI: false,
};

function App() {
  return (
    <div>
      {FEATURES.newDashboard && <NewDashboard />}
      {FEATURES.betaFeature && <BetaFeature />}

      <MainContent />
    </div>
  );
}
```

---

**下一步**: 学习 [列表和 Keys](./lists-and-keys) 了解如何渲染列表，或查看 [Hooks 详解](./hooks) 深入学习 React。
