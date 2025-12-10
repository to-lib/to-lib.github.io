---
sidebar_position: 22
title: TypeScript 与 React
---

# TypeScript 与 React

> [!TIP]
> TypeScript 为 React 应用提供了强大的类型安全保障。本文涵盖 React + TypeScript 的核心用法。

## 🚀 快速开始

### 创建 TypeScript 项目

```bash
# 使用 Vite
npm create vite@latest my-app -- --template react-ts

# 使用 Create React App
npx create-react-app my-app --template typescript
```

## 📝 组件类型

### 函数组件

```tsx
// 基础组件
function Greeting(): JSX.Element {
  return <h1>Hello</h1>;
}

// 带 Props
interface GreetingProps {
  name: string;
  age?: number; // 可选
}

function Greeting({ name, age }: GreetingProps): JSX.Element {
  return <h1>Hello {name}</h1>;
}

// 使用 FC 类型（可选）
const Greeting: React.FC<GreetingProps> = ({ name, age }) => {
  return <h1>Hello {name}</h1>;
};
```

### Children Props

```tsx
interface CardProps {
  title: string;
  children: React.ReactNode;
}

function Card({ title, children }: CardProps) {
  return (
    <div>
      <h2>{title}</h2>
      {children}
    </div>
  );
}
```

## 🎯 Hooks 类型

### useState

```tsx
// 类型推断
const [count, setCount] = useState(0); // number

// 显式类型
const [user, setUser] = useState<User | null>(null);
const [items, setItems] = useState<string[]>([]);

interface User {
  id: number;
  name: string;
}
```

### useRef

```tsx
// DOM 引用
const inputRef = useRef<HTMLInputElement>(null);

// 可变值
const timerRef = useRef<number | null>(null);
```

### useReducer

```tsx
interface State {
  count: number;
}

type Action =
  | { type: "increment" }
  | { type: "decrement" }
  | { type: "set"; payload: number };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "increment":
      return { count: state.count + 1 };
    case "decrement":
      return { count: state.count - 1 };
    case "set":
      return { count: action.payload };
    default:
      return state;
  }
}

function Counter() {
  const [state, dispatch] = useReducer(reducer, { count: 0 });

  return (
    <div>
      <p>{state.count}</p>
      <button onClick={() => dispatch({ type: "increment" })}>+</button>
    </div>
  );
}
```

## 🎨 事件处理

```tsx
function Form() {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
  };

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    console.log(e.currentTarget);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input onChange={handleChange} />
      <button onClick={handleClick}>Submit</button>
    </form>
  );
}
```

## 💡 常用类型

```tsx
// 组件 Props
interface ButtonProps {
  variant: "primary" | "secondary";
  size?: "small" | "medium" | "large";
  onClick?: () => void;
  children: React.ReactNode;
}

// 表单事件
type InputChangeEvent = React.ChangeEvent<HTMLInputElement>;
type FormSubmitEvent = React.FormEvent<HTMLFormElement>;

// 样式对象
const styles: React.CSSProperties = {
  color: "red",
  fontSize: 16,
};
```

## 🔧 高级类型

### 泛型组件

```tsx
interface ListProps<T> {
  items: T[];
  renderItem: (item: T) => React.ReactNode;
}

function List<T>({ items, renderItem }: ListProps<T>) {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={index}>{renderItem(item)}</li>
      ))}
    </ul>
  );
}

// 使用
<List<User> items={users} renderItem={(user) => <div>{user.name}</div>} />;
```

### forwardRef

```tsx
interface InputProps {
  placeholder?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ placeholder }, ref) => {
    return <input ref={ref} placeholder={placeholder} />;
  }
);
```

## 📚 实用示例

```tsx
// API 响应类型
interface User {
  id: number;
  name: string;
  email: string;
}

interface ApiResponse<T> {
  data: T;
  error?: string;
}

// 数据获取组件
function UserProfile({ userId }: { userId: number }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/api/users/${userId}`)
      .then((res) => res.json() as Promise<ApiResponse<User>>)
      .then(({ data }) => {
        setUser(data);
        setLoading(false);
      });
  }, [userId]);

  if (loading) return <div>Loading...</div>;
  if (!user) return <div>Not found</div>;

  return <div>{user.name}</div>;
}
```

---

**下一步**：学习 [测试](/docs/react/testing) 保证代码质量，或查看 [SSR/Next.js](/docs/react/ssr-nextjs) 了解服务端渲染。
