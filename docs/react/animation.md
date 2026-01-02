---
sidebar_position: 21
title: 动画
---

# React 动画解决方案

> [!TIP]
> React 提供多种动画方案：CSS 动画、Framer Motion、React Spring 等。选择合适的方案可以大幅提升用户体验。

## 🎯 动画方案对比

| 方案                | 优点               | 适用场景       |
| ------------------- | ------------------ | -------------- |
| **CSS Transitions** | 简单、性能好       | 简单过渡       |
| **CSS Animations**  | 无需 JS、性能好    | 复杂关键帧动画 |
| **Framer Motion**   | API 优雅、功能强大 | 大多数场景推荐 |
| **React Spring**    | 物理动画、自然效果 | 弹性效果       |
| **GSAP**            | 专业级、时间线控制 | 复杂交互动画   |

## 🎨 CSS 动画

### 过渡动画

```jsx
function FadeButton() {
  const [isVisible, setIsVisible] = useState(true);

  return (
    <div>
      <button onClick={() => setIsVisible(!isVisible)}>Toggle</button>
      <div
        style={{
          opacity: isVisible ? 1 : 0,
          transition: "opacity 0.3s ease",
        }}
      >
        内容
      </div>
    </div>
  );
}
```

### CSS Modules 动画

```css
/* styles.module.css */
.fadeIn {
  animation: fadeIn 0.3s ease-in;
}

.fadeOut {
  animation: fadeOut 0.3s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
```

```jsx
import styles from "./styles.module.css";

function AnimatedCard({ isVisible }) {
  return (
    <div className={isVisible ? styles.fadeIn : styles.fadeOut}>
      Card Content
    </div>
  );
}
```

## ⚡ Framer Motion（推荐）

Framer Motion 是 React 最流行的动画库。

### 安装

```bash
npm install framer-motion
```

### 基础动画

```jsx
import { motion } from "framer-motion";

function AnimatedBox() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      Hello Motion
    </motion.div>
  );
}
```

### 交互动画

```jsx
function InteractiveCard() {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      style={{
        padding: 20,
        background: "#fff",
        borderRadius: 8,
        cursor: "pointer",
      }}
    >
      Hover and tap me!
    </motion.div>
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
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="modal-overlay"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            transition={{ type: "spring", damping: 25 }}
            className="modal-content"
            onClick={(e) => e.stopPropagation()}
          >
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
```

### 列表动画

```jsx
function AnimatedList({ items }) {
  return (
    <ul>
      {items.map((item, index) => (
        <motion.li
          key={item.id}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          {item.name}
        </motion.li>
      ))}
    </ul>
  );
}
```

### 布局动画

```jsx
function LayoutAnimation() {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <motion.div
      layout
      onClick={() => setIsExpanded(!isExpanded)}
      style={{
        width: isExpanded ? 300 : 100,
        height: isExpanded ? 200 : 100,
        background: "#3b82f6",
        borderRadius: 16,
      }}
    />
  );
}
```

### 手势拖拽

```jsx
function DraggableCard() {
  return (
    <motion.div
      drag
      dragConstraints={{ top: -50, left: -50, right: 50, bottom: 50 }}
      whileDrag={{ scale: 1.1 }}
      style={{
        width: 100,
        height: 100,
        background: "#10b981",
        borderRadius: 16,
        cursor: "grab",
      }}
    />
  );
}
```

## 🌊 React Spring

React Spring 提供物理效果的动画。

### 安装

```bash
npm install @react-spring/web
```

### 基础用法

```jsx
import { useSpring, animated } from "@react-spring/web";

function FadeIn() {
  const springs = useSpring({
    from: { opacity: 0 },
    to: { opacity: 1 },
  });

  return <animated.div style={springs}>Hello Spring</animated.div>;
}
```

### 交互动画

```jsx
function HoverCard() {
  const [springs, api] = useSpring(() => ({
    scale: 1,
    config: { tension: 300, friction: 10 },
  }));

  return (
    <animated.div
      onMouseEnter={() => api.start({ scale: 1.1 })}
      onMouseLeave={() => api.start({ scale: 1 })}
      style={{
        transform: springs.scale.to((s) => `scale(${s})`),
        width: 100,
        height: 100,
        background: "#8b5cf6",
        borderRadius: 16,
      }}
    />
  );
}
```

### 数字动画

```jsx
function AnimatedNumber({ value }) {
  const { number } = useSpring({
    from: { number: 0 },
    number: value,
    config: { mass: 1, tension: 20, friction: 10 },
  });

  return <animated.span>{number.to((n) => n.toFixed(0))}</animated.span>;
}

// 使用
<AnimatedNumber value={1000} />;
```

## 🎭 Transition Group

React Transition Group 是底层动画库，用于进入/退出动画。

### 安装

```bash
npm install react-transition-group
```

### 使用 CSSTransition

```jsx
import { CSSTransition } from "react-transition-group";
import "./fade.css";

function FadeWrapper({ show, children }) {
  return (
    <CSSTransition in={show} timeout={300} classNames="fade" unmountOnExit>
      {children}
    </CSSTransition>
  );
}
```

```css
/* fade.css */
.fade-enter {
  opacity: 0;
}
.fade-enter-active {
  opacity: 1;
  transition: opacity 300ms;
}
.fade-exit {
  opacity: 1;
}
.fade-exit-active {
  opacity: 0;
  transition: opacity 300ms;
}
```

## 💡 性能优化

### 1. 使用 transform 和 opacity

```jsx
// ✅ 好：只动画 transform 和 opacity（GPU 加速）
<motion.div animate={{ x: 100, opacity: 0.5 }} />

// ❌ 避免：动画 width、height、top、left（触发重排）
<motion.div animate={{ width: 200, top: 100 }} />
```

### 2. 使用 will-change

```css
.animated-element {
  will-change: transform, opacity;
}
```

### 3. 减少动画的 DOM 节点

```jsx
// 使用 layout="position" 只动画位置变化
<motion.div layout="position" />
```

### 4. 尊重用户偏好

```jsx
import { useReducedMotion } from "framer-motion";

function AnimatedComponent() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <motion.div
      animate={{ x: 100 }}
      transition={{ duration: shouldReduceMotion ? 0 : 0.5 }}
    />
  );
}
```

## 🔗 相关资源

- [Framer Motion 官方文档](https://www.framer.com/motion/)
- [React Spring 官方文档](https://www.react-spring.dev/)
- [可访问性](/docs/react/accessibility) - 动画与无障碍
- [性能优化](/docs/react/performance-optimization)

---

**下一步**：了解 [可访问性](/docs/react/accessibility) 确保动画对所有用户友好。
