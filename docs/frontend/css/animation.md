---
sidebar_position: 5
title: 动画与过渡
---

# CSS 动画与过渡

> [!TIP]
> CSS 动画让网页元素动起来，提升用户体验和视觉吸引力。

## 🎬 过渡 (Transition)

过渡用于在状态变化时添加平滑效果：

```css
.button {
  background: #3b82f6;
  transition: background 0.3s ease;
}

.button:hover {
  background: #1d4ed8;
}
```

### 语法

```css
transition: property duration timing-function delay;

/* 示例 */
transition: all 0.3s ease 0s;
transition: transform 0.5s ease-in-out;
transition: opacity 0.2s, transform 0.3s;
```

### 属性说明

| 属性                         | 说明     | 常用值                          |
| ---------------------------- | -------- | ------------------------------- |
| `transition-property`        | 过渡属性 | `all`, `transform`, `opacity`   |
| `transition-duration`        | 持续时间 | `0.3s`, `200ms`                 |
| `transition-timing-function` | 缓动函数 | `ease`, `linear`, `ease-in-out` |
| `transition-delay`           | 延迟时间 | `0s`, `0.1s`                    |

### 缓动函数

```css
/* 预设值 */
transition-timing-function: ease; /* 默认，慢-快-慢 */
transition-timing-function: linear; /* 匀速 */
transition-timing-function: ease-in; /* 慢入 */
transition-timing-function: ease-out; /* 慢出 */
transition-timing-function: ease-in-out; /* 慢入慢出 */

/* 自定义贝塞尔曲线 */
transition-timing-function: cubic-bezier(0.68, -0.55, 0.265, 1.55);
```

## 🎨 动画 (Animation)

动画可以创建更复杂的多帧效果：

### @keyframes 定义

```css
@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

@keyframes bounce {
  0%,
  100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-20px);
  }
}

@keyframes pulse {
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
  100% {
    transform: scale(1);
  }
}
```

### 应用动画

```css
.element {
  animation: fadeIn 1s ease-out;
}

/* 完整语法 */
animation: name duration timing-function delay iteration-count direction
  fill-mode;

/* 示例 */
animation: bounce 0.5s ease-in-out infinite;
animation: fadeIn 0.3s ease-out forwards;
```

### 动画属性

| 属性                        | 说明     | 常用值                           |
| --------------------------- | -------- | -------------------------------- |
| `animation-name`            | 动画名称 | `@keyframes` 定义的名称          |
| `animation-duration`        | 持续时间 | `1s`, `500ms`                    |
| `animation-timing-function` | 缓动函数 | `ease`, `linear`                 |
| `animation-delay`           | 延迟     | `0s`, `0.5s`                     |
| `animation-iteration-count` | 重复次数 | `1`, `3`, `infinite`             |
| `animation-direction`       | 播放方向 | `normal`, `reverse`, `alternate` |
| `animation-fill-mode`       | 结束状态 | `none`, `forwards`, `backwards`  |
| `animation-play-state`      | 播放状态 | `running`, `paused`              |

## 💫 常用动画示例

### 淡入效果

```css
@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.fade-in {
  animation: fadeIn 0.5s ease-out;
}
```

### 滑入效果

```css
@keyframes slideIn {
  from {
    transform: translateX(-100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.slide-in {
  animation: slideIn 0.3s ease-out;
}
```

### 旋转加载

```css
@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}
```

### 脉冲效果

```css
@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.pulse {
  animation: pulse 2s ease-in-out infinite;
}
```

## ⚡ 性能优化

### 使用 transform 和 opacity

```css
/* ✅ 性能好 - 只触发合成 */
.good {
  transform: translateX(100px);
  opacity: 0.5;
}

/* ❌ 性能差 - 触发重排 */
.bad {
  left: 100px;
  width: 200px;
}
```

### will-change 提示

```css
.animated {
  will-change: transform, opacity;
}

/* 动画结束后移除 */
.animated.done {
  will-change: auto;
}
```

### 减少动画范围

```css
/* 只在需要时启用动画 */
@media (prefers-reduced-motion: no-preference) {
  .element {
    animation: fadeIn 0.3s ease-out;
  }
}

/* 用户偏好减少动画 */
@media (prefers-reduced-motion: reduce) {
  .element {
    animation: none;
  }
}
```

## 💡 最佳实践

### 1. 保持动画简短

```css
/* ✅ 推荐：300ms 以内 */
transition: transform 0.2s ease;

/* ❌ 过长会让用户等待 */
transition: transform 2s ease;
```

### 2. 使用合适的缓动

```css
/* 进入动画 - ease-out */
.enter {
  animation: slideIn 0.3s ease-out;
}

/* 离开动画 - ease-in */
.leave {
  animation: slideOut 0.2s ease-in;
}
```

### 3. 避免闪烁

```css
.element {
  backface-visibility: hidden;
  transform: translateZ(0);
}
```

## 🔗 相关资源

- [CSS 入门](/docs/frontend/css/)
- [响应式设计](/docs/frontend/css/responsive)

---

**下一步**：学习 [JavaScript 基础](/docs/frontend/javascript/) 添加交互动画。
