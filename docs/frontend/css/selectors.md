---
sidebar_position: 2
title: 选择器
---

# CSS 选择器

> [!TIP]
> 选择器决定样式应用到哪些元素。掌握选择器是写好 CSS 的关键。

## 🎯 基础选择器

### 元素选择器

```css
p {
  color: black;
}
h1 {
  font-size: 2em;
}
a {
  text-decoration: none;
}
```

### 类选择器

```css
.highlight {
  background: yellow;
}
.btn {
  padding: 10px 20px;
}
.btn.primary {
  background: blue;
} /* 多个类 */
```

```html
<p class="highlight">高亮文本</p>
<button class="btn primary">按钮</button>
```

### ID 选择器

```css
#header {
  height: 60px;
}
#main-content {
  padding: 20px;
}
```

### 通用选择器

```css
* {
  margin: 0;
  padding: 0;
}
```

## 🔗 组合选择器

### 后代选择器（空格）

```css
/* 所有后代 */
.card p {
  margin: 10px;
}

article h2 {
  color: blue;
}
```

### 子选择器（>）

```css
/* 直接子元素 */
.menu > li {
  display: inline-block;
}
```

### 相邻兄弟选择器（+）

```css
/* 紧邻的下一个兄弟 */
h1 + p {
  font-size: 1.2em;
}
```

### 通用兄弟选择器（~）

```css
/* 后面所有兄弟 */
h1 ~ p {
  color: gray;
}
```

## 📝 属性选择器

```css
/* 有该属性 */
[disabled] {
  opacity: 0.5;
}

/* 属性等于值 */
[type="text"] {
  border: 1px solid #ccc;
}

/* 属性包含值（空格分隔的列表） */
[class~="btn"] {
  cursor: pointer;
}

/* 属性以值开头 */
[href^="https"] {
  color: green;
}

/* 属性以值结尾 */
[href$=".pdf"] {
  color: red;
}

/* 属性包含值 */
[href*="example"] {
  font-weight: bold;
}
```

## 🎭 伪类选择器

### 状态伪类

```css
/* 悬停 */
a:hover {
  color: red;
}

/* 激活（点击时） */
button:active {
  transform: scale(0.95);
}

/* 获得焦点 */
input:focus {
  border-color: blue;
}

/* 已访问链接 */
a:visited {
  color: purple;
}
```

### 结构伪类

```css
/* 第一个/最后一个子元素 */
li:first-child {
  font-weight: bold;
}
li:last-child {
  border-bottom: none;
}

/* 第 n 个子元素 */
li:nth-child(2) {
  color: red;
} /* 第2个 */
li:nth-child(odd) {
  background: #f0f0f0;
} /* 奇数行 */
li:nth-child(even) {
  background: #fff;
} /* 偶数行 */
li:nth-child(3n) {
  color: blue;
} /* 每隔3个 */

/* 唯一子元素 */
p:only-child {
  font-style: italic;
}

/* 空元素 */
div:empty {
  display: none;
}
```

### 表单伪类

```css
input:disabled {
  background: #eee;
}
input:enabled {
  background: #fff;
}
input:checked {
  outline: 2px solid blue;
}
input:required {
  border-left: 3px solid red;
}
input:valid {
  border-color: green;
}
input:invalid {
  border-color: red;
}
input::placeholder {
  color: #999;
}
```

### 否定伪类

```css
/* 排除某些元素 */
p:not(.special) {
  color: gray;
}
input:not([type="submit"]) {
  width: 100%;
}
```

## ✨ 伪元素选择器

```css
/* 首字母 */
p::first-letter {
  font-size: 2em;
  float: left;
}

/* 首行 */
p::first-line {
  font-weight: bold;
}

/* 选中文本 */
::selection {
  background: yellow;
  color: black;
}

/* 生成内容 */
.required::before {
  content: "*";
  color: red;
}

.external-link::after {
  content: " ↗";
}
```

### before/after 实用示例

```css
/* 清除浮动 */
.clearfix::after {
  content: "";
  display: block;
  clear: both;
}

/* 装饰性元素 */
.card::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(to right, blue, purple);
}

/* 图标 */
.download::before {
  content: "📥 ";
}
```

## ⚖️ 选择器优先级

### 优先级权重

| 选择器       | 权重 |
| ------------ | ---- |
| `!important` | 最高 |
| 内联样式     | 1000 |
| ID 选择器    | 100  |
| 类/伪类/属性 | 10   |
| 元素/伪元素  | 1    |
| 通用选择器   | 0    |

### 计算示例

```css
/* 权重: 0-0-1 = 1 */
p {
}

/* 权重: 0-1-0 = 10 */
.text {
}

/* 权重: 1-0-0 = 100 */
#main {
}

/* 权重: 0-1-1 = 11 */
p.text {
}

/* 权重: 1-1-1 = 111 */
#main p.text {
}
```

### 优先级规则

```css
/* 同优先级，后面覆盖前面 */
p {
  color: red;
}
p {
  color: blue;
} /* 生效 */

/* 高优先级覆盖低优先级 */
p {
  color: red;
} /* 权重 1 */
.text {
  color: blue;
} /* 权重 10，生效 */

/* !important 最高优先级（慎用） */
p {
  color: red !important;
}
```

## 💡 最佳实践

### 1. 避免过度具体

```css
/* ❌ 太具体 */
div.container ul.menu li.item a.link {
}

/* ✅ 简洁 */
.menu-link {
}
```

### 2. 使用类而非 ID

```css
/* ❌ ID 不可复用 */
#submit-button {
}

/* ✅ 类可复用 */
.btn-submit {
}
```

### 3. 避免 !important

```css
/* ❌ 避免 */
.button {
  color: blue !important;
}

/* ✅ 提高特异性 */
.form .button {
  color: blue;
}
```

## 🔗 相关资源

- [CSS 入门](/docs/frontend/css/)
- [布局](/docs/frontend/css/layout)
- [响应式设计](/docs/frontend/css/responsive)

---

**下一步**：学习 [布局](/docs/frontend/css/layout) 掌握 Flexbox 和 Grid。
