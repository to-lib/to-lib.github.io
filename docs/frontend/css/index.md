---
sidebar_position: 1
title: CSS 入门
---

# CSS 基础

> [!TIP]
> CSS（Cascading Style Sheets）用于控制网页的视觉表现，包括颜色、布局、字体等。

## 🎯 什么是 CSS？

CSS 定义了 HTML 元素如何显示：

```css
/* 选择器 { 属性: 值; } */
h1 {
  color: blue;
  font-size: 24px;
}
```

## 📦 引入方式

### 1. 外部样式表（推荐）

```html
<head>
  <link rel="stylesheet" href="styles.css" />
</head>
```

### 2. 内部样式表

```html
<head>
  <style>
    h1 {
      color: blue;
    }
  </style>
</head>
```

### 3. 内联样式

```html
<h1 style="color: blue; font-size: 24px;">标题</h1>
```

## 🎨 基础语法

### 选择器

```css
/* 元素选择器 */
p {
  color: black;
}

/* 类选择器 */
.highlight {
  background: yellow;
}

/* ID 选择器 */
#header {
  height: 60px;
}

/* 组合 */
.card p {
  margin: 10px;
}
```

### 属性和值

```css
selector {
  property: value;
  property: value;
}
```

## 🎨 常用属性

### 颜色

```css
/* 文字颜色 */
color: red;
color: #ff0000;
color: rgb(255, 0, 0);
color: rgba(255, 0, 0, 0.5);
color: hsl(0, 100%, 50%);

/* 背景颜色 */
background-color: #f0f0f0;
```

### 文字

```css
font-family: "Arial", sans-serif;
font-size: 16px;
font-weight: bold; /* normal, bold, 100-900 */
font-style: italic;
text-align: center; /* left, right, center, justify */
text-decoration: none; /* underline, line-through */
line-height: 1.5;
letter-spacing: 1px;
```

### 尺寸

```css
width: 100px;
width: 50%;
width: 100vw; /* 视口宽度 */
max-width: 1200px;
min-width: 320px;

height: 200px;
height: 100vh; /* 视口高度 */
```

### 内外边距

```css
/* 外边距 */
margin: 10px; /* 四边 */
margin: 10px 20px; /* 上下 左右 */
margin: 10px 20px 30px 40px; /* 上 右 下 左 */
margin-top: 10px;

/* 内边距 */
padding: 10px;
padding: 10px 20px;
```

### 边框

```css
border: 1px solid black;
border-radius: 8px; /* 圆角 */
border-radius: 50%; /* 圆形 */
```

### 背景

```css
background-color: #fff;
background-image: url("image.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;

/* 简写 */
background: #fff url("image.jpg") center/cover no-repeat;
```

## 📦 盒模型

```
┌─────────────────────────────────┐
│            margin               │
│   ┌───────────────────────┐     │
│   │       border          │     │
│   │  ┌─────────────────┐  │     │
│   │  │    padding      │  │     │
│   │  │  ┌───────────┐  │  │     │
│   │  │  │  content  │  │  │     │
│   │  │  └───────────┘  │  │     │
│   │  └─────────────────┘  │     │
│   └───────────────────────┘     │
└─────────────────────────────────┘
```

```css
/* 默认盒模型 */
box-sizing: content-box; /* width/height 只包含 content */

/* 推荐盒模型 */
box-sizing: border-box; /* width/height 包含 padding + border */

/* 全局设置 */
* {
  box-sizing: border-box;
}
```

## 🎭 显示和定位

### display

```css
display: block; /* 块级元素 */
display: inline; /* 行内元素 */
display: inline-block; /* 行内块 */
display: none; /* 隐藏 */
display: flex; /* Flexbox */
display: grid; /* Grid */
```

### position

```css
position: static; /* 默认 */
position: relative; /* 相对定位 */
position: absolute; /* 绝对定位 */
position: fixed; /* 固定定位 */
position: sticky; /* 粘性定位 */

top: 10px;
right: 10px;
bottom: 10px;
left: 10px;
z-index: 100;
```

### 定位示例

```css
/* 相对定位 - 相对于自身原位置 */
.relative {
  position: relative;
  top: 10px;
  left: 20px;
}

/* 绝对定位 - 相对于最近的定位祖先 */
.parent {
  position: relative;
}
.child {
  position: absolute;
  top: 0;
  right: 0;
}

/* 固定定位 - 相对于视口 */
.fixed-header {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
}
```

## 💡 最佳实践

### 1. 使用 border-box

```css
*,
*::before,
*::after {
  box-sizing: border-box;
}
```

### 2. CSS 重置

```css
* {
  margin: 0;
  padding: 0;
}

body {
  font-family: system-ui, -apple-system, sans-serif;
  line-height: 1.5;
}
```

### 3. 使用 CSS 变量

```css
:root {
  --primary-color: #3b82f6;
  --text-color: #333;
  --spacing: 16px;
}

.button {
  background: var(--primary-color);
  padding: var(--spacing);
}
```

## 🔗 相关资源

- [选择器详解](/docs/frontend/css/selectors)
- [布局](/docs/frontend/css/layout)
- [响应式设计](/docs/frontend/css/responsive)

---

**下一步**：学习 [选择器](/docs/frontend/css/selectors) 掌握元素选择技巧。
