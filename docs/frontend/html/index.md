---
sidebar_position: 1
title: HTML 入门
---

# HTML 基础

> [!TIP]
> HTML（HyperText Markup Language）是网页的骨架，定义了网页的结构和内容。

## 🎯 什么是 HTML？

HTML 是一种**标记语言**，用于描述网页的结构：

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>我的第一个网页</title>
  </head>
  <body>
    <h1>Hello World!</h1>
    <p>这是我的第一个网页。</p>
  </body>
</html>
```

## 📦 文档结构

### DOCTYPE 声明

```html
<!DOCTYPE html>
```

- 告诉浏览器使用 HTML5 标准解析
- 必须放在文档最开头

### html 元素

```html
<html lang="zh-CN">
  <!-- 整个网页内容 -->
</html>
```

- `lang="zh-CN"` 声明页面语言（有助于 SEO 和辅助技术）

### head 元素

```html
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="页面描述" />
  <title>页面标题</title>
  <link rel="stylesheet" href="style.css" />
</head>
```

| 元素                 | 作用           |
| -------------------- | -------------- |
| `<meta charset>`     | 字符编码       |
| `<meta viewport>`    | 移动端适配     |
| `<meta description>` | SEO 描述       |
| `<title>`            | 浏览器标签标题 |
| `<link>`             | 引入外部资源   |

### body 元素

```html
<body>
  <header>头部</header>
  <main>主要内容</main>
  <footer>底部</footer>
</body>
```

## 🏷️ 标签语法

### 双标签

```html
<标签名>内容</标签名>

<!-- 示例 -->
<p>这是一个段落</p>
<div>这是一个容器</div>
```

### 单标签（自闭合）

```html
<标签名 />

<!-- 示例 -->
<br />
<!-- 换行 -->
<hr />
<!-- 水平线 -->
<img src="image.jpg" alt="图片" />
<input type="text" />
```

### 属性

```html
<标签名 属性名="属性值">内容</标签名>

<!-- 示例 -->
<a href="https://example.com" target="_blank">链接</a>
<img src="photo.jpg" alt="照片" width="200" />
```

## 📝 常用元素速览

### 文本

```html
<h1>一级标题</h1>
<h2>二级标题</h2>
<p>段落文本</p>
<span>行内文本</span>
<strong>加粗</strong>
<em>斜体</em>
```

### 链接和图片

```html
<a href="https://example.com">链接文本</a>
<img src="image.jpg" alt="图片描述" />
```

### 列表

```html
<!-- 无序列表 -->
<ul>
  <li>项目 1</li>
  <li>项目 2</li>
</ul>

<!-- 有序列表 -->
<ol>
  <li>第一步</li>
  <li>第二步</li>
</ol>
```

### 容器

```html
<div>块级容器</div>
<span>行内容器</span>
```

## 🎨 注释

```html
<!-- 这是注释，不会显示在页面上 -->

<!--
  多行注释
  可以写多行
-->
```

## 💡 最佳实践

### 1. 使用正确的文档结构

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>页面标题</title>
  </head>
  <body>
    <!-- 内容 -->
  </body>
</html>
```

### 2. 标签嵌套正确

```html
<!-- ✅ 正确 -->
<p><strong>加粗文本</strong></p>

<!-- ❌ 错误 -->
<p><strong>加粗文本</p></strong>
```

### 3. 属性使用双引号

```html
<!-- ✅ 推荐 -->
<img src="image.jpg" alt="描述" />

<!-- ❌ 不推荐 -->
<img src="image.jpg" alt="描述" />
```

### 4. 图片添加 alt 属性

```html
<!-- ✅ 有 alt -->
<img src="photo.jpg" alt="风景照片" />

<!-- ❌ 无 alt -->
<img src="photo.jpg" />
```

## 🔗 相关资源

- [常用元素](/docs/frontend/html/elements)
- [表单](/docs/frontend/html/forms)
- [语义化](/docs/frontend/html/semantic)

---

**下一步**：学习 [常用元素](/docs/frontend/html/elements) 了解更多 HTML 标签。
