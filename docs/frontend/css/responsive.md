---
sidebar_position: 4
title: 响应式设计
---

# CSS 响应式设计

> [!TIP]
> 响应式设计让网页在不同设备上都能良好显示，从手机到桌面显示器。

## 🎯 核心概念

### 视口设置

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
```

### 移动优先

```css
/* 默认样式（移动端） */
.container {
  padding: 16px;
}

/* 平板 */
@media (min-width: 768px) {
  .container {
    padding: 24px;
  }
}

/* 桌面 */
@media (min-width: 1024px) {
  .container {
    padding: 32px;
    max-width: 1200px;
  }
}
```

## 📱 媒体查询

### 基础语法

```css
@media (条件) {
  /* 样式规则 */
}

/* 最小宽度 */
@media (min-width: 768px) {
}

/* 最大宽度 */
@media (max-width: 767px) {
}

/* 范围 */
@media (min-width: 768px) and (max-width: 1023px) {
}
```

### 常用断点

```css
/* 手机 */
@media (max-width: 639px) {
}

/* 平板 */
@media (min-width: 640px) {
}

/* 小电脑 */
@media (min-width: 768px) {
}

/* 电脑 */
@media (min-width: 1024px) {
}

/* 大屏 */
@media (min-width: 1280px) {
}
```

### 其他媒体特性

```css
/* 横屏/竖屏 */
@media (orientation: landscape) {
}
@media (orientation: portrait) {
}

/* 暗色模式 */
@media (prefers-color-scheme: dark) {
  body {
    background: #1a1a1a;
    color: #fff;
  }
}

/* 减少动效 */
@media (prefers-reduced-motion: reduce) {
  * {
    animation: none !important;
    transition: none !important;
  }
}
```

## 📐 响应式单位

### 相对单位

```css
/* 相对于父元素字体大小 */
font-size: 1.5em;

/* 相对于根元素字体大小 */
font-size: 1rem;

/* 视口单位 */
width: 100vw; /* 视口宽度 */
height: 100vh; /* 视口高度 */
font-size: 5vw; /* 视口宽度的5% */

/* 容器单位（CSS Container Queries） */
width: 50cqw; /* 容器宽度的50% */
```

### 百分比

```css
.container {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
}

.column {
  width: 50%;
}
```

### clamp() 函数

```css
/* clamp(最小值, 理想值, 最大值) */
font-size: clamp(1rem, 2.5vw, 2rem);
width: clamp(320px, 90%, 1200px);
padding: clamp(16px, 4vw, 48px);
```

## 🖼️ 响应式图片

### 基础响应式

```css
img {
  max-width: 100%;
  height: auto;
}
```

### srcset

```html
<img
  src="image-800.jpg"
  srcset="image-400.jpg 400w, image-800.jpg 800w, image-1200.jpg 1200w"
  sizes="
    (max-width: 400px) 100vw,
    (max-width: 800px) 50vw,
    33vw
  "
  alt="响应式图片"
/>
```

### picture 元素

```html
<picture>
  <source media="(min-width: 1024px)" srcset="desktop.jpg" />
  <source media="(min-width: 640px)" srcset="tablet.jpg" />
  <img src="mobile.jpg" alt="图片" />
</picture>
```

## 📦 响应式布局模式

### 流式布局

```css
.container {
  width: 90%;
  max-width: 1200px;
  margin: 0 auto;
}
```

### Flexbox 响应式

```css
.cards {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}

.card {
  flex: 1 1 300px; /* 最小300px，自动伸缩 */
}
```

### Grid 响应式

```css
/* 自动填充 */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 24px;
}

/* 媒体查询切换 */
.layout {
  display: grid;
  grid-template-columns: 1fr;
}

@media (min-width: 768px) {
  .layout {
    grid-template-columns: 200px 1fr;
  }
}

@media (min-width: 1024px) {
  .layout {
    grid-template-columns: 200px 1fr 200px;
  }
}
```

## 📝 响应式排版

```css
html {
  font-size: 16px;
}

@media (min-width: 768px) {
  html {
    font-size: 18px;
  }
}

/* 流体排版 */
h1 {
  font-size: clamp(1.5rem, 4vw, 3rem);
}

p {
  font-size: clamp(1rem, 2vw, 1.25rem);
  line-height: 1.6;
}
```

## 📱 移动端优化

### 触控友好

```css
/* 最小点击区域 44x44px */
button,
a {
  min-height: 44px;
  min-width: 44px;
  padding: 12px;
}

/* 间距加大 */
.nav-link + .nav-link {
  margin-left: 16px;
}
```

### 隐藏/显示元素

```css
/* 移动端隐藏 */
.desktop-only {
  display: none;
}

@media (min-width: 768px) {
  .desktop-only {
    display: block;
  }

  .mobile-only {
    display: none;
  }
}
```

## 💡 完整示例

```css
/* 基础样式（移动优先） */
* {
  box-sizing: border-box;
}

body {
  font-family: system-ui, sans-serif;
  line-height: 1.6;
  padding: 16px;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.nav {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.cards {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}

/* 平板 */
@media (min-width: 640px) {
  body {
    padding: 24px;
  }

  .header {
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
  }

  .nav {
    flex-direction: row;
  }

  .cards {
    grid-template-columns: repeat(2, 1fr);
    gap: 24px;
  }
}

/* 桌面 */
@media (min-width: 1024px) {
  .cards {
    grid-template-columns: repeat(3, 1fr);
  }
}
```

## 🔗 相关资源

- [CSS 入门](/docs/frontend/css/)
- [布局](/docs/frontend/css/layout)
- [JavaScript 入门](/docs/frontend/javascript/)

---

**下一步**：学习 [JavaScript 入门](/docs/frontend/javascript/) 为网页添加交互。
