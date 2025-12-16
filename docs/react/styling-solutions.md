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
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.secondary {
  background: #6c757d;
}

.large {
  padding: 14px 28px;
  font-size: 16px;
}
```

```jsx
import styles from "./Button.module.css";
import clsx from "clsx"; // 推荐使用 clsx 合并类名

function Button({ variant = "primary", size, children, ...props }) {
  return (
    <button
      className={clsx(styles.button, styles[variant], size && styles[size])}
      {...props}
    >
      {children}
    </button>
  );
}
```

## 🌊 Tailwind CSS（推荐）

### 安装配置

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

```js
// tailwind.config.js
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          50: "#eff6ff",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
        },
      },
    },
  },
  plugins: [],
};
```

### 组件示例

```jsx
function Button({ variant = "primary", children }) {
  const baseStyles =
    "px-4 py-2 rounded-lg font-medium transition-all duration-200";
  const variants = {
    primary:
      "bg-primary-500 hover:bg-primary-600 text-white shadow-lg hover:shadow-xl",
    secondary: "bg-gray-100 hover:bg-gray-200 text-gray-800",
    outline: "border-2 border-primary-500 text-primary-500 hover:bg-primary-50",
  };

  return (
    <button className={`${baseStyles} ${variants[variant]}`}>{children}</button>
  );
}
```

### 响应式设计

```jsx
<div
  className="
  w-full
  md:w-1/2      /* 中等屏幕 */
  lg:w-1/3      /* 大屏幕 */
  p-4
  md:p-6
  lg:p-8
"
>
  <h1 className="text-xl md:text-2xl lg:text-4xl font-bold">响应式标题</h1>
</div>
```

## 💅 Styled Components

```bash
npm install styled-components
npm install -D @types/styled-components  # TypeScript
```

### 基础用法

```jsx
import styled from "styled-components";

const StyledButton = styled.button`
  padding: 10px 20px;
  background: ${(props) => (props.$primary ? "#3b82f6" : "#6c757d")};
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

function Button({ primary, children, ...props }) {
  return (
    <StyledButton $primary={primary} {...props}>
      {children}
    </StyledButton>
  );
}
```

### 主题配置

```jsx
import { ThemeProvider, createGlobalStyle } from "styled-components";

const theme = {
  colors: {
    primary: "#3b82f6",
    secondary: "#6c757d",
    background: "#ffffff",
    text: "#1f2937",
  },
  spacing: {
    sm: "8px",
    md: "16px",
    lg: "24px",
  },
  borderRadius: "8px",
};

const GlobalStyle = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Inter', sans-serif;
    background: ${(props) => props.theme.colors.background};
    color: ${(props) => props.theme.colors.text};
  }
`;

function App() {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <YourApp />
    </ThemeProvider>
  );
}
```

## 😊 Emotion

```bash
npm install @emotion/react @emotion/styled
```

### CSS-in-JS

```jsx
/** @jsxImportSource @emotion/react */
import { css } from "@emotion/react";
import styled from "@emotion/styled";

// 使用 css 属性
const buttonStyles = css`
  padding: 10px 20px;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;

  &:hover {
    background: #2563eb;
  }
`;

function Button({ children }) {
  return <button css={buttonStyles}>{children}</button>;
}

// 使用 styled 组件
const Card = styled.div`
  padding: ${(props) => props.theme.spacing.md};
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
`;
```

## 🎬 动画方案 (Framer Motion)

```bash
npm install framer-motion
```

### 基础动画

```jsx
import { motion } from "framer-motion";

function AnimatedButton({ children }) {
  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      transition={{ type: "spring", stiffness: 400, damping: 17 }}
    >
      {children}
    </motion.button>
  );
}
```

### 进入/退出动画

```jsx
import { motion, AnimatePresence } from "framer-motion";

function Modal({ isOpen, onClose, children }) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            className="overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.div
            className="modal"
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: "spring", duration: 0.5 }}
          >
            {children}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
```

### 列表动画

```jsx
function AnimatedList({ items }) {
  return (
    <motion.ul>
      {items.map((item, index) => (
        <motion.li
          key={item.id}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          {item.text}
        </motion.li>
      ))}
    </motion.ul>
  );
}
```

## 🎨 CSS 变量方案

```css
/* globals.css */
:root {
  /* 颜色 */
  --color-primary: #3b82f6;
  --color-primary-hover: #2563eb;
  --color-secondary: #6c757d;
  --color-background: #ffffff;
  --color-text: #1f2937;

  /* 间距 */
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;

  /* 圆角 */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;

  /* 阴影 */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

/* 暗色主题 */
[data-theme="dark"] {
  --color-background: #1f2937;
  --color-text: #f9fafb;
  --color-primary: #60a5fa;
}
```

```jsx
// 主题切换
function ThemeToggle() {
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <button onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
      切换主题
    </button>
  );
}

// 使用变量
const Button = styled.button`
  background: var(--color-primary);
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-md);

  &:hover {
    background: var(--color-primary-hover);
  }
`;
```

## 💡 选择建议

| 场景     | 推荐方案                                   |
| -------- | ------------------------------------------ |
| 新项目   | **Tailwind CSS** - 开发效率高              |
| 组件库   | **CSS Modules** - 无运行时开销             |
| 主题切换 | **Styled Components / Emotion** - 动态主题 |
| 动画需求 | **Framer Motion** + 任意 CSS 方案          |
| 性能敏感 | **CSS Modules** / **Tailwind** - 零运行时  |

---

**相关文档**：[项目结构](/docs/react/project-structure) | [最佳实践](/docs/react/best-practices)
