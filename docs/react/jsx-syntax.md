---
sidebar_position: 4
title: JSX 语法
---

# JSX 语法详解

> [!TIP]
> JSX 是 JavaScript 的语法扩展，让你可以在 JavaScript 中编写类似 HTML 的代码。

## 📚 什么是 JSX？

JSX (JavaScript XML) 是 React 创建元素的语法糖，会被编译成 `React.createElement()` 调用。

### JSX vs JavaScript

```jsx
// JSX 写法
const element = <h1 className="greeting">Hello, world!</h1>;

// 编译后的 JavaScript
const element = React.createElement(
  "h1",
  { className: "greeting" },
  "Hello, world!"
);
```

## 🎯 JSX 基础语法

### 1. 嵌入表达式

使用 `{}` 在 JSX 中嵌入 JavaScript 表达式：

```jsx
const name = "Alice";
const age = 25;

const element = (
  <div>
    <h1>Hello, {name}!</h1>
    <p>You are {age} years old</p>
    <p>Next year you'll be {age + 1}</p>
  </div>
);
```

### 2. 属性使用

```jsx
// 字符串属性
<img src="avatar.jpg" alt="User Avatar" />

// 表达式属性
const imageUrl = 'https://example.com/image.jpg';
<img src={imageUrl} alt="Dynamic" />

// 注意：class 要写成 className
<div className="container">Content</div>

// style 使用对象
<div style={{ color: 'red', fontSize: '16px' }}>
  Styled Text
</div>
```

### 3. 子元素

```jsx
// 单个子元素
<div>
  <h1>Title</h1>
</div>

// 多个子元素
<div>
  <h1>Title</h1>
  <p>Paragraph</p>
  <button>Click</button>
</div>

// 嵌入表达式作为子元素
<ul>
  {items.map(item => <li key={item.id}>{item.name}</li>)}
</ul>
```

## ⚠️ JSX 规则

### 1. 必须有一个根元素

```jsx
// ✗ 错误 - 多个根元素
function Component() {
  return (
    <h1>Title</h1>
    <p>Paragraph</p>
  );
}

// ✓ 正确 - 使用 Fragment
function Component() {
  return (
    <>
      <h1>Title</h1>
      <p>Paragraph</p>
    </>
  );
}

// 或使用 div
function Component() {
  return (
    <div>
      <h1>Title</h1>
      <p>Paragraph</p>
    </div>
  );
}
```

### 2. 标签必须闭合

```jsx
// ✗ 错误
<img src="image.jpg">
<input type="text">

// ✓ 正确
<img src="image.jpg" />
<input type="text" />
```

### 3. 使用 camelCase 命名

```jsx
// HTML 属性名
<div class="container" onclick="handleClick()">

// JSX 属性名（camelCase）
<div className="container" onClick={handleClick}>
```

## 🔤 JSX 中的 JavaScript

### 1. 条件渲染

```jsx
// 三元运算符
function Greeting({ isLoggedIn }) {
  return (
    <div>{isLoggedIn ? <h1>Welcome back!</h1> : <h1>Please sign in</h1>}</div>
  );
}

// 逻辑与 &&
function Inbox({ unreadCount }) {
  return (
    <div>
      <h1>Messages</h1>
      {unreadCount > 0 && <p>You have {unreadCount} unread messages</p>}
    </div>
  );
}
```

### 2. 列表渲染

```jsx
function TodoList({ todos }) {
  return (
    <ul>
      {todos.map((todo) => (
        <li key={todo.id}>{todo.text}</li>
      ))}
    </ul>
  );
}
```

### 3. 函数调用

```jsx
function formatName(user) {
  return `${user.firstName} ${user.lastName}`;
}

function Greeting({ user }) {
  return <h1>Hello, {formatName(user)}!</h1>;
}
```

## 🎨 JSX 属性详解

### className 和 style

```jsx
// className (注意不是 class)
<div className="container primary">Content</div>

// 动态 className
const isActive = true;
<div className={isActive ? 'active' : 'inactive'}>Item</div>

// style 对象
<div style={{
  color: 'blue',
  backgroundColor: 'lightgray',
  fontSize: '16px',
  padding: '10px'
}}>
  Styled Content
</div>

// 提取 style 对象
const styles = {
  container: {
    color: 'blue',
    padding: '10px'
  }
};
<div style={styles.container}>Content</div>
```

### 事件处理

```jsx
function Button() {
  const handleClick = () => {
    alert("Button clicked!");
  };

  return <button onClick={handleClick}>Click Me</button>;
}

// 传递参数
function List({ items }) {
  const handleDelete = (id) => {
    console.log("Delete", id);
  };

  return (
    <ul>
      {items.map((item) => (
        <li key={item.id}>
          {item.name}
          <button onClick={() => handleDelete(item.id)}>Delete</button>
        </li>
      ))}
    </ul>
  );
}
```

### 表单属性

```jsx
function Form() {
  const [value, setValue] = useState("");

  return (
    <form>
      {/* htmlFor 代替 for */}
      <label htmlFor="name">Name:</label>
      <input
        id="name"
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />

      {/* defaultValue 用于非受控组件 */}
      <input type="text" defaultValue="Initial" />
    </form>
  );
}
```

## 🔍 特殊用法

### Fragment

不添加额外 DOM 节点的包装器：

```jsx
// 完整语法
import { Fragment } from "react";

function List() {
  return (
    <Fragment>
      <li>Item 1</li>
      <li>Item 2</li>
    </Fragment>
  );
}

// 简写语法
function List() {
  return (
    <>
      <li>Item 1</li>
      <li>Item 2</li>
    </>
  );
}

// 带 key 的 Fragment（必须用完整语法）
function Glossary({ items }) {
  return (
    <dl>
      {items.map((item) => (
        <Fragment key={item.id}>
          <dt>{item.term}</dt>
          <dd>{item.description}</dd>
        </Fragment>
      ))}
    </dl>
  );
}
```

### 注释

```jsx
function Component() {
  return (
    <div>
      {/* 这是 JSX 中的注释 */}
      <h1>Title</h1>

      {/* 
        多行注释
        也可以这样写
      */}
      <p>Content</p>
    </div>
  );
}
```

### 展开运算符

```jsx
const props = {
  name: 'Alice',
  age: 25,
  email: 'alice@example.com'
};

// 使用展开运算符传递所有 props
<UserCard {...props} />

// 等同于
<UserCard name="Alice" age={25} email="alice@example.com" />

// 覆盖某些属性
<UserCard {...props} age={26} />
```

## 💡 JSX 最佳实践

### 1. 保持代码可读

```jsx
// ✗ 不好 - 太长
<Button onClick={handleClick} className="primary large" disabled={isLoading} aria-label="Submit form" />

// ✓ 好 - 多行格式
<Button
  onClick={handleClick}
  className="primary large"
  disabled={isLoading}
  aria-label="Submit form"
/>
```

### 2. 避免嵌套过深

```jsx
// ✗ 不好
return (
  <div>
    <div>
      <div>
        <div>
          <h1>Too Deep</h1>
        </div>
      </div>
    </div>
  </div>
);

// ✓ 好 - 提取组件
function Header() {
  return <h1>Better</h1>;
}

return (
  <div>
    <Header />
  </div>
);
```

### 3. 条件渲染简洁化

```jsx
// ✗ 不好
{
  condition ? <Component /> : null;
}

// ✓ 好
{
  condition && <Component />;
}

// 复杂条件提取
const shouldShow = isLoggedIn && hasPermission && !isLoading;
{
  shouldShow && <Component />;
}
```

### 4. 使用常量存储复杂 JSX

```jsx
function Profile({ user }) {
  const userInfo = (
    <div className="user-info">
      <h2>{user.name}</h2>
      <p>{user.email}</p>
    </div>
  );

  const userActions = (
    <div className="actions">
      <button>Edit</button>
      <button>Delete</button>
    </div>
  );

  return (
    <div className="profile">
      {userInfo}
      {userActions}
    </div>
  );
}
```

## 🔧 编译过程

### Babel 转换

```jsx
// 源代码
const element = <h1 className="greeting">Hello, {name}!</h1>;

// Babel 转换后 (React 17 之前)
const element = React.createElement(
  "h1",
  { className: "greeting" },
  "Hello, ",
  name,
  "!"
);

// React 17+ 新 JSX 转换
import { jsx as _jsx } from "react/jsx-runtime";

const element = _jsx("h1", {
  className: "greeting",
  children: ["Hello, ", name, "!"],
});
```

## 📖 实用示例

### 动态类名

```jsx
function Button({ primary, disabled, children }) {
  const classes = ["btn", primary && "btn-primary", disabled && "btn-disabled"]
    .filter(Boolean)
    .join(" ");

  return <button className={classes}>{children}</button>;
}

// 或使用 classnames 库
import classNames from "classnames";

function Button({ primary, disabled, children }) {
  return (
    <button
      className={classNames("btn", {
        "btn-primary": primary,
        "btn-disabled": disabled,
      })}
    >
      {children}
    </button>
  );
}
```

### 表单示例

```jsx
function ContactForm() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
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
        type="text"
        name="name"
        value={formData.name}
        onChange={handleChange}
        placeholder="Your Name"
      />
      <input
        type="email"
        name="email"
        value={formData.email}
        onChange={handleChange}
        placeholder="Your Email"
      />
      <textarea
        name="message"
        value={formData.message}
        onChange={handleChange}
        placeholder="Your Message"
      />
      <button type="submit">Send</button>
    </form>
  );
}
```

---

**下一步**: 了解 [Props 和 State](/docs/react/props-and-state) 学习组件数据管理，或查看 [事件处理](/docs/react/event-handling) 了解用户交互。
