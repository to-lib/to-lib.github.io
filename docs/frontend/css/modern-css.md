---
sidebar_position: 6
title: CSS 新特性
---

# CSS 新特性

> [!TIP]
> CSS 在不断进化，新特性让我们能用更少的代码实现更强大的布局和样式。

## 📦 容器查询 (Container Queries)

响应容器大小而非视口大小，让组件真正可复用。

### 基础用法

```css
/* 定义容器 */
.card-container {
  container-type: inline-size;
  container-name: card;
}

/* 根据容器宽度调整样式 */
@container card (min-width: 400px) {
  .card {
    display: flex;
    flex-direction: row;
  }
}

@container card (max-width: 399px) {
  .card {
    display: block;
  }
}
```

### 简写语法

```css
.container {
  container: card / inline-size;
}

/* 匿名容器查询 */
@container (min-width: 300px) {
  .item {
    font-size: 1.2rem;
  }
}
```

### 容器查询单位

```css
.card-title {
  /* 容器宽度的百分比 */
  font-size: 5cqw;

  /* 容器高度的百分比 */
  padding: 2cqh;

  /* 容器较小尺寸的百分比 */
  margin: 1cqmin;
}
```

## 🎯 :has() 选择器

"父级选择器"，根据子元素选择父元素。

### 基础用法

```css
/* 包含图片的卡片 */
.card:has(img) {
  display: grid;
  grid-template-columns: 200px 1fr;
}

/* 没有图片的卡片 */
.card:not(:has(img)) {
  padding: 2rem;
}
```

### 实用示例

```css
/* 表单验证状态 */
.form-group:has(input:invalid) {
  border-color: red;
}

.form-group:has(input:valid) {
  border-color: green;
}

/* 悬停卡片时改变子元素 */
.card:has(.card-link:hover) {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* 折叠面板打开时 */
details:has([open]) summary {
  color: blue;
}
```

### 兄弟选择

```css
/* 选择有焦点输入框的相邻标签 */
label:has(+ input:focus) {
  color: blue;
  font-weight: bold;
}
```

## 🎨 cascade layers

控制样式的优先级，更好地组织 CSS。

```css
/* 定义层级顺序 */
@layer reset, base, components, utilities;

/* reset 层 - 最低优先级 */
@layer reset {
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
}

/* base 层 */
@layer base {
  body {
    font-family: system-ui;
    line-height: 1.5;
  }
}

/* components 层 */
@layer components {
  .btn {
    padding: 0.5rem 1rem;
    border-radius: 4px;
  }
}

/* utilities 层 - 最高优先级 */
@layer utilities {
  .hidden {
    display: none !important;
  }
}
```

## 🎭 嵌套语法 (Nesting)

原生 CSS 支持嵌套，类似 Sass。

```css
.card {
  padding: 1rem;
  background: white;

  /* 嵌套选择器 */
  & .title {
    font-size: 1.5rem;
    font-weight: bold;
  }

  & .content {
    color: #666;
  }

  /* 伪类 */
  &:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }

  /* 媒体查询嵌套 */
  @media (min-width: 768px) {
    padding: 2rem;
  }
}
```

## 🔧 Subgrid

子网格继承父网格的轨道定义。

```css
.grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
}

.grid-item {
  display: grid;
  /* 继承父网格的列 */
  grid-template-columns: subgrid;
  /* 跨越两列 */
  grid-column: span 2;
}
```

## ⚡ 新的颜色函数

### color-mix()

```css
.button {
  background: blue;
}

.button:hover {
  /* 混合 80% 原色和 20% 白色 */
  background: color-mix(in srgb, blue 80%, white);
}

.button:active {
  /* 混合原色和黑色 */
  background: color-mix(in srgb, blue, black 20%);
}
```

### oklch() 和 oklab()

更均匀的颜色空间，适合创建调色板。

```css
:root {
  /* 基础色 */
  --primary: oklch(60% 0.15 250);

  /* 更亮的变体 */
  --primary-light: oklch(75% 0.15 250);

  /* 更暗的变体 */
  --primary-dark: oklch(45% 0.15 250);
}
```

## 📦 新的视口单位

更准确的移动端视口处理。

```css
.hero {
  /* 最小视口高度 - 适合移动端 */
  min-height: 100svh;
}

.fixed-bottom {
  /* 最大视口高度 */
  bottom: calc(100lvh - 100%);
}

.modal {
  /* 动态视口高度 */
  max-height: 100dvh;
}
```

| 单位      | 说明                        |
| --------- | --------------------------- |
| `svh/svw` | Small viewport - 最小视口   |
| `lvh/lvw` | Large viewport - 最大视口   |
| `dvh/dvw` | Dynamic viewport - 动态变化 |

## 🎮 scroll-driven animations

基于滚动的动画，无需 JavaScript。

```css
@keyframes fade-in {
  from {
    opacity: 0;
    transform: translateY(50px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.card {
  animation: fade-in linear both;
  /* 滚动时播放动画 */
  animation-timeline: view();
  /* 进入视口时开始 */
  animation-range: entry 0% cover 40%;
}
```

## 💡 浏览器支持检测

```css
/* 检测是否支持某特性 */
@supports (container-type: inline-size) {
  .container {
    container-type: inline-size;
  }
}

/* 不支持时的回退 */
@supports not (container-type: inline-size) {
  .container {
    /* 使用媒体查询作为回退 */
  }
}
```

## 🔗 相关资源

- [CSS 入门](/docs/frontend/css/)
- [响应式设计](/docs/frontend/css/responsive)
- [CSS 动画](/docs/frontend/css/animation)

---

**下一步**：学习 [Web Workers](/docs/frontend/browser/workers) 了解多线程处理。
