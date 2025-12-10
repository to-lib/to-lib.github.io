---
sidebar_position: 91
title: 最佳实践
---

# React 最佳实践

> [!TIP]
> 遵循最佳实践能让你的 React 代码更清晰、可维护、高性能。

## 📁 项目结构

### 推荐的目录组织

```
src/
├── components/       # 可复用组件
│   ├── Button/
│   │   ├── Button.tsx
│   │   ├── Button.test.tsx
│   │   └── index.ts
├── features/         # 功能模块
│   ├── auth/
│   ├── dashboard/
├── hooks/            # 自定义 Hooks
├── utils/            # 工具函数
├── types/            # TypeScript 类型
├── styles/           # 全局样式
└── App.tsx
```

## 🎯 组件设计原则

### 1. 单一职责

```jsx
// ✓ 好：职责清晰
function UserAvatar({ imageUrl, size }) {
  return <img src={imageUrl} width={size} />;
}

function UserName({ name }) {
  return <span>{name}</span>;
}

// ✗ 不好：混杂多个职责
function UserCard({ user }) {
  return (
    <div>
      <img src={user.avatar} />
      <span>{user.name}</span>
      <button onClick={handleDelete}>删除</button>
      <form>{/* 编辑表单 */}</form>
    </div>
  );
}
```

### 2. 组合优于继承

```jsx
// ✓ 好：使用组合
function Card({ children }) {
  return <div className="card">{children}</div>;
}

<Card>
  <UserInfo user={user} />
  <Actions />
</Card>;

// ✗ 不好：使用继承
class BaseCard extends React.Component {}
class UserCard extends BaseCard {}
```

### 3. Props 解构

```jsx
// ✓ 好：解构 Props
function Button({ variant, size, children, ...rest }) {
  return (
    <button className={`btn-${variant}-${size}`} {...rest}>
      {children}
    </button>
  );
}

// ✗ 不好：直接使用 props
function Button(props) {
  return <button className={`btn-${props.variant}`}>{props.children}</button>;
}
```

## ⚡ 性能优化

### 1. 避免不必要的渲染

```jsx
// ✓ 使用 React.memo
const ExpensiveComponent = React.memo(({ data }) => {
  return <div>{/* 复杂渲染 */}</div>;
});

// ✓ 使用 useMemo 缓存计算结果
const sortedData = useMemo(
  () => data.sort((a, b) => a.value - b.value),
  [data]
);

// ✓ 使用 useCallback 缓存函数
const handleClick = useCallback(() => {
  doSomething();
}, []);
```

### 2. 合理使用 Key

```jsx
// ✓ 好：使用稳定的 ID
{
  items.map((item) => <Item key={item.id} {...item} />);
}

// ✗ 不好：使用索引（列表会变化时）
{
  items.map((item, index) => <Item key={index} {...item} />);
}
```

## 🔒 状态管理

### 1. 状态放置位置

```jsx
// ✓ 好：状态下放到需要的组件
function Parent() {
  return <Child />; // Parent 无需关心 Child 的状态
}

function Child() {
  const [open, setOpen] = useState(false);
  return <Modal open={open} />;
}

// ✗ 不好：状态提升过高
function Parent() {
  const [childOpen, setChildOpen] = useState(false);
  return <Child open={childOpen} setOpen={setChildOpen} />;
}
```

### 2. 状态扁平化

```jsx
// ✓ 好：扁平的状态结构
const [firstName, setFirstName] = useState("");
const [lastName, setLastName] = useState("");

// ✗ 不好：过深的嵌套
const [user, setUser] = useState({
  profile: {
    name: { first: "", last: "" },
  },
});
```

## 📝 代码风格

### 1. 使用 TypeScript

```tsx
// ✓ 定义明确的类型
interface UserProps {
  id: number;
  name: string;
  email?: string;
}

function User({ id, name, email }: UserProps) {
  // ...
}
```

### 2. 命名规范

```jsx
// 组件名：PascalCase
function UserProfile() {}

// Hooks：use 开头
function useAuth() {}

// 事件处理：handle 开头
const handleClick = () => {};

// 布尔值：is/has 开头
const isLoading = true;
const hasError = false;
```

### 3. 避免魔法数字

```jsx
// ✓ 好：使用常量
const MAX_ITEMS = 10;
const DEBOUNCE_DELAY = 500;

// ✗ 不好：直接使用数字
setTimeout(() => {}, 500);
items.slice(0, 10);
```

## 🧪 测试

```jsx
// ✓ 测试用户行为
it("shows modal when clicking button", async () => {
  render(<App />);
  await userEvent.click(screen.getByText("Open"));
  expect(screen.getByRole("dialog")).toBeInTheDocument();
});

// ✗ 测试实现细节
it("sets state correctly", () => {
  // 不推荐
});
```

## 🔐 安全

```jsx
// ✓ 好：转义用户输入
<div>{sanitize(userInput)}</div>

// ✗ 危险：dangerouslySetInnerHTML
<div dangerouslySetInnerHTML={{ __html: userInput }} />

// ✓ 好：验证外部链接
<a href={url} rel="noopener noreferrer" target="_blank">
```

## 💡 通用建议

1. **保持组件小而专一**
2. **使用自定义 Hooks 复用逻辑**
3. **优先使用函数组件和 Hooks**
4. **合理使用 TypeScript**
5. **编写测试保证质量**
6. **遵循可访问性标准**
7. **定期更新依赖**

---

**相关资源**：[快速参考](/docs/react/quick-reference) | [性能优化](/docs/react/performance-optimization) | [FAQ](/docs/react/faq)
