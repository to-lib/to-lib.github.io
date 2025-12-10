---
sidebar_position: 26
title: CSS 方案
---

# React 样式解决方案

> [!TIP]
> React 有多种样式方案可选。本文对比常见方案帮助你做出选择。

## 📊 方案对比

| 方案                  | 学习曲线 | 性能 | 推荐度     |
| --------------------- | -------- | ---- | ---------- |
| **CSS Modules**       | 低       | 高   | ⭐⭐⭐⭐   |
| **Tailwind CSS**      | 中       | 高   | ⭐⭐⭐⭐⭐ |
| **Styled Components** | 中       | 中   | ⭐⭐⭐     |
| **Emotion**           | 中       | 中   | ⭐⭐⭐     |

## 🎨 CSS Modules

```css
/* Button.module.css */
.button {
  padding: 10px 20px;
  background: blue;
  color: white;
}

.primary {
  background: green;
}
```

```jsx
import styles from "./Button.module.css";

function Button({ variant }) {
  return (
    <button
      className={`${styles.button} ${
        variant === "primary" ? styles.primary : ""
      }`}
    >
      Click me
    </button>
  );
}
```

## 🌊 Tailwind CSS（推荐）

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init
```

```jsx
function Button({ variant }) {
  return (
    <button
      className={`
      px-4 py-2 rounded
      ${variant === "primary" ? "bg-blue-500" : "bg-gray-500"}
      text-white hover:opacity-80
    `}
    >
      Click me
    </button>
  );
}
```

## 💅 Styled Components

```bash
npm install styled-components
```

```jsx
import styled from "styled-components";

const StyledButton = styled.button`
  padding: 10px 20px;
  background: ${(props) => (props.primary ? "blue" : "gray")};
  color: white;
  border: none;

  &:hover {
    opacity: 0.8;
  }
`;

function Button({ primary }) {
  return <StyledButton primary={primary}>Click me</StyledButton>;
}
```

---

**选择建议**：新项目推荐使用 **Tailwind CSS**，组件库推荐 **CSS Modules**。
