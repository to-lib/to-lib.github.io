---
sidebar_position: 23
title: 测试
---

# React 应用测试

> [!TIP]
> 测试是保证代码质量的重要手段。本文介绍 React 应用的单元测试、集成测试和 E2E 测试。

## 📦 测试工具

| 工具                      | 用途       | 推荐场景   |
| ------------------------- | ---------- | ---------- |
| **Vitest**                | 测试运行器 | Vite 项目  |
| **Jest**                  | 测试运行器 | 传统项目   |
| **React Testing Library** | 组件测试   | 所有项目   |
| **Playwright**            | E2E 测试   | 端到端测试 |

## 🚀 快速开始

### 安装（Vitest + React Testing Library）

```bash
npm install -D vitest @testing-library/react @testing-library/jest-dom
npm install -D @testing-library/user-event jsdom
```

### 配置

```javascript
// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
});
```

```typescript
// src/test/setup.ts
import "@testing-library/jest-dom";
```

## 🧪 组件测试

### 基础测试

```tsx
// Button.tsx
interface ButtonProps {
  onClick: () => void;
  children: React.ReactNode;
}

export function Button({ onClick, children }: ButtonProps) {
  return <button onClick={onClick}>{children}</button>;
}

// Button.test.tsx
import { render, screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { Button } from "./Button";

describe("Button", () => {
  it("renders children correctly", () => {
    render(<Button onClick={() => {}}>Click me</Button>);
    expect(screen.getByText("Click me")).toBeInTheDocument();
  });

  it("calls onClick when clicked", async () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click me</Button>);

    await userEvent.click(screen.getByRole("button"));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
```

### 测试异步组件

```tsx
// UserProfile.tsx
function UserProfile({ userId }: { userId: number }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/api/users/${userId}`)
      .then((res) => res.json())
      .then((data) => {
        setUser(data);
        setLoading(false);
      });
  }, [userId]);

  if (loading) return <div>Loading...</div>;
  return <div>{user?.name}</div>;
}

// UserProfile.test.tsx
describe("UserProfile", () => {
  it("shows loading state initially", () => {
    render(<UserProfile userId={1} />);
    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("displays user name after loading", async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ name: "John" }),
      })
    );

    render(<UserProfile userId={1} />);

    expect(await screen.findByText("John")).toBeInTheDocument();
  });
});
```

### 测试 Hooks

```tsx
// useCounter.ts
export function useCounter(initialValue = 0) {
  const [count, setCount] = useState(initialValue);
  const increment = () => setCount((c) => c + 1);
  const decrement = () => setCount((c) => c - 1);
  return { count, increment, decrement };
}

// useCounter.test.ts
import { renderHook, act } from "@testing-library/react";

describe("useCounter", () => {
  it("increments counter", () => {
    const { result } = renderHook(() => useCounter(0));

    act(() => {
      result.current.increment();
    });

    expect(result.current.count).toBe(1);
  });
});
```

## 💡 最佳实践

### 1. 测试用户行为，而非实现

```tsx
// ✗ 不好：测试实现细节
it("sets state to true", () => {
  const { result } = renderHook(() => useState(false));
  act(() => result.current[1](true));
  expect(result.current[0]).toBe(true);
});

// ✓ 好：测试用户可见的行为
it("shows modal when button is clicked", async () => {
  render(<App />);
  await userEvent.click(screen.getByText("Open Modal"));
  expect(screen.getByRole("dialog")).toBeInTheDocument();
});
```

### 2. 使用可访问性查询

```tsx
// ✓ 推荐：使用语义化查询
screen.getByRole("button", { name: "Submit" });
screen.getByLabelText("Email");
screen.getByPlaceholderText("Enter name");

// ✗ 避免：使用实现细节
screen.getByClassName("submit-btn");
screen.getByTestId("email-input");
```

---

**下一步**：学习 [代码分割](./code-splitting) 优化加载性能，或查看 [最佳实践](./best-practices) 掌握开发规范。
