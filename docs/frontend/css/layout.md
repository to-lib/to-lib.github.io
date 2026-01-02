---
sidebar_position: 3
title: 布局
---

# CSS 布局

> [!TIP]
> 现代 CSS 布局主要使用 Flexbox 和 Grid，它们让复杂布局变得简单。

## 🎯 Flexbox

Flexbox 是一维布局方案，适合在一行或一列中排列元素。

### 基础概念

```css
.container {
  display: flex;
}
```

```
     主轴 (main axis) →
   ┌────────────────────────┐
   │ item1 │ item2 │ item3  │  ↓ 交叉轴 (cross axis)
   └────────────────────────┘
```

### 容器属性

```css
.container {
  display: flex;

  /* 主轴方向 */
  flex-direction: row; /* 水平（默认） */
  flex-direction: row-reverse;
  flex-direction: column; /* 垂直 */
  flex-direction: column-reverse;

  /* 换行 */
  flex-wrap: nowrap; /* 不换行（默认） */
  flex-wrap: wrap; /* 换行 */

  /* 主轴对齐 */
  justify-content: flex-start; /* 起点 */
  justify-content: flex-end; /* 终点 */
  justify-content: center; /* 居中 */
  justify-content: space-between; /* 两端对齐 */
  justify-content: space-around; /* 等间距 */
  justify-content: space-evenly; /* 完全等分 */

  /* 交叉轴对齐 */
  align-items: stretch; /* 拉伸（默认） */
  align-items: flex-start;
  align-items: flex-end;
  align-items: center;

  /* 多行对齐 */
  align-content: flex-start;
  align-content: center;
  align-content: space-between;

  /* 间距 */
  gap: 10px;
  row-gap: 10px;
  column-gap: 20px;
}
```

### 项目属性

```css
.item {
  /* 放大比例 */
  flex-grow: 1; /* 占满剩余空间 */

  /* 缩小比例 */
  flex-shrink: 0; /* 不缩小 */

  /* 基础尺寸 */
  flex-basis: 200px;
  flex-basis: 30%;

  /* 简写 */
  flex: 1; /* flex-grow: 1 */
  flex: 1 0 200px; /* grow shrink basis */

  /* 单独对齐 */
  align-self: center;

  /* 排序 */
  order: 1;
}
```

### Flexbox 常用布局

```css
/* 水平垂直居中 */
.center {
  display: flex;
  justify-content: center;
  align-items: center;
}

/* 等分布局 */
.equal-columns {
  display: flex;
}
.equal-columns > * {
  flex: 1;
}

/* 圣杯布局 */
.holy-grail {
  display: flex;
}
.holy-grail .sidebar {
  flex: 0 0 200px;
}
.holy-grail .main {
  flex: 1;
}

/* 底部固定 */
.sticky-footer {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
.sticky-footer main {
  flex: 1;
}
```

## 📐 Grid

Grid 是二维布局方案，适合同时控制行和列。

### 基础概念

```css
.container {
  display: grid;
  grid-template-columns: 200px 1fr 200px;
  grid-template-rows: auto 1fr auto;
}
```

```
   ┌─────────┬────────────────┬─────────┐
   │ header  │     header     │ header  │
   ├─────────┼────────────────┼─────────┤
   │ sidebar │     main       │  aside  │
   ├─────────┼────────────────┼─────────┤
   │ footer  │     footer     │ footer  │
   └─────────┴────────────────┴─────────┘
```

### 容器属性

```css
.container {
  display: grid;

  /* 定义列 */
  grid-template-columns: 100px 200px 100px;
  grid-template-columns: 1fr 2fr 1fr; /* 比例 */
  grid-template-columns: repeat(3, 1fr); /* 重复 */
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); /* 自动填充 */

  /* 定义行 */
  grid-template-rows: 60px 1fr 40px;

  /* 间距 */
  gap: 20px;
  row-gap: 10px;
  column-gap: 20px;

  /* 命名区域 */
  grid-template-areas:
    "header header header"
    "sidebar main aside"
    "footer footer footer";

  /* 对齐 */
  justify-items: center; /* 水平对齐 */
  align-items: center; /* 垂直对齐 */
  place-items: center; /* 简写 */

  justify-content: center; /* 整体水平对齐 */
  align-content: center; /* 整体垂直对齐 */
}
```

### 项目属性

```css
.item {
  /* 指定区域 */
  grid-area: header;

  /* 指定位置 */
  grid-column: 1 / 3; /* 第1到第3列 */
  grid-row: 1 / 2;

  /* 简写 */
  grid-column: span 2; /* 跨2列 */
  grid-row: 2 / -1; /* 从第2行到最后 */

  /* 单独对齐 */
  justify-self: end;
  align-self: start;
}
```

### Grid 常用布局

```css
/* 等分网格 */
.grid-equal {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

/* 响应式卡片 */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 24px;
}

/* 圣杯布局 */
.holy-grail-grid {
  display: grid;
  grid-template-columns: 200px 1fr 200px;
  grid-template-rows: auto 1fr auto;
  grid-template-areas:
    "header header header"
    "nav main aside"
    "footer footer footer";
  min-height: 100vh;
}
.header {
  grid-area: header;
}
.nav {
  grid-area: nav;
}
.main {
  grid-area: main;
}
.aside {
  grid-area: aside;
}
.footer {
  grid-area: footer;
}
```

## ⚖️ Flexbox vs Grid

| 特性     | Flexbox        | Grid               |
| -------- | -------------- | ------------------ |
| 维度     | 一维           | 二维               |
| 适用场景 | 导航、卡片排列 | 页面布局、复杂网格 |
| 内容驱动 | ✅             | ❌                 |
| 布局驱动 | ❌             | ✅                 |

```css
/* Flexbox: 内容决定布局 */
.nav {
  display: flex;
  gap: 10px;
}

/* Grid: 布局决定内容位置 */
.page {
  display: grid;
  grid-template-columns: 200px 1fr;
}
```

## 🔗 相关资源

- [CSS 入门](/docs/frontend/css/)
- [选择器](/docs/frontend/css/selectors)
- [响应式设计](/docs/frontend/css/responsive)

---

**下一步**：学习 [响应式设计](/docs/frontend/css/responsive) 适配不同设备。
