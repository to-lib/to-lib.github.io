---
sidebar_position: 4
title: 语义化
---

# HTML 语义化

> [!TIP]
> 语义化 HTML 让代码更有意义，提升可访问性、SEO 和代码可维护性。

## 🎯 什么是语义化？

语义化是使用**有意义的标签**来描述内容，而不是只用 `<div>` 和 `<span>`。

```html
<!-- ❌ 非语义化 -->
<div class="header">
  <div class="nav">...</div>
</div>
<div class="main">
  <div class="article">...</div>
</div>
<div class="footer">...</div>

<!-- ✅ 语义化 -->
<header>
  <nav>...</nav>
</header>
<main>
  <article>...</article>
</main>
<footer>...</footer>
```

## 📦 结构元素

### header

```html
<!-- 页面头部 -->
<header>
  <h1>网站名称</h1>
  <nav>导航菜单</nav>
</header>

<!-- 文章头部 -->
<article>
  <header>
    <h2>文章标题</h2>
    <time datetime="2024-01-01">2024年1月1日</time>
  </header>
  <p>文章内容...</p>
</article>
```

### nav

```html
<nav>
  <ul>
    <li><a href="/">首页</a></li>
    <li><a href="/about">关于</a></li>
    <li><a href="/contact">联系</a></li>
  </ul>
</nav>
```

### main

```html
<main>
  <!-- 页面主要内容，每页只有一个 -->
  <h1>页面标题</h1>
  <article>...</article>
</main>
```

### article

```html
<article>
  <!-- 独立完整的内容单元 -->
  <h2>文章标题</h2>
  <p>文章内容...</p>
</article>
```

### section

```html
<section>
  <!-- 主题相关的内容分组 -->
  <h2>章节标题</h2>
  <p>章节内容...</p>
</section>
```

### aside

```html
<aside>
  <!-- 侧边栏、广告、相关链接 -->
  <h3>相关文章</h3>
  <ul>
    <li><a href="#">文章1</a></li>
    <li><a href="#">文章2</a></li>
  </ul>
</aside>
```

### footer

```html
<footer>
  <p>&copy; 2024 网站名称</p>
  <nav>
    <a href="/privacy">隐私政策</a>
    <a href="/terms">服务条款</a>
  </nav>
</footer>
```

## 🏗️ 完整页面结构

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <title>语义化示例</title>
  </head>
  <body>
    <header>
      <h1>网站名称</h1>
      <nav>
        <ul>
          <li><a href="/">首页</a></li>
          <li><a href="/blog">博客</a></li>
        </ul>
      </nav>
    </header>

    <main>
      <article>
        <header>
          <h2>文章标题</h2>
          <time datetime="2024-01-01">2024年1月1日</time>
        </header>

        <section>
          <h3>第一章</h3>
          <p>内容...</p>
        </section>

        <section>
          <h3>第二章</h3>
          <p>内容...</p>
        </section>

        <footer>
          <p>作者：张三</p>
        </footer>
      </article>

      <aside>
        <h3>相关文章</h3>
        <ul>
          <li><a href="#">相关文章1</a></li>
        </ul>
      </aside>
    </main>

    <footer>
      <p>&copy; 2024 版权所有</p>
    </footer>
  </body>
</html>
```

## 📝 文本语义

```html
<!-- 时间 -->
<time datetime="2024-01-01T10:00:00">2024年1月1日 10:00</time>

<!-- 地址 -->
<address>
  联系我们：<a href="mailto:hello@example.com">hello@example.com</a>
</address>

<!-- 缩写 -->
<abbr title="HyperText Markup Language">HTML</abbr>

<!-- 引用 -->
<blockquote cite="https://example.com">这是一段引用文本。</blockquote>

<!-- 代码 -->
<code>console.log('Hello')</code>
<pre><code>function hello() {
  return 'world';
}</code></pre>

<!-- 定义 -->
<dfn>HTML</dfn> 是超文本标记语言。

<!-- 标记/高亮 -->
<p>搜索结果中的 <mark>关键词</mark> 会被高亮。</p>
```

## 🎯 为什么要语义化？

### 1. 可访问性

屏幕阅读器可以正确理解页面结构：

```html
<!-- 屏幕阅读器知道这是导航 -->
<nav>...</nav>

<!-- 只知道这是一个div -->
<div class="nav">...</div>
```

### 2. SEO 优化

搜索引擎更好理解内容：

```html
<!-- 搜索引擎知道这是主要内容 -->
<article>
  <h1>重要标题</h1>
  <p>正文内容...</p>
</article>
```

### 3. 代码可读性

```html
<!-- 一眼就知道结构 -->
<header>...</header>
<main>
  <article>...</article>
  <aside>...</aside>
</main>
<footer>...</footer>
```

## 💡 最佳实践

### 1. 使用正确的标题层级

```html
<!-- ✅ 正确 -->
<h1>页面标题</h1>
<h2>章节标题</h2>
<h3>子章节</h3>

<!-- ❌ 跳过层级 -->
<h1>页面标题</h1>
<h4>子标题</h4>
```

### 2. 每个页面只有一个 h1

```html
<h1>页面主标题</h1>
```

### 3. 每个页面只有一个 main

```html
<main>
  <!-- 页面核心内容 -->
</main>
```

### 4. 使用语义元素替代 div

| 用途     | 使用        | 而非                    |
| -------- | ----------- | ----------------------- |
| 页面头部 | `<header>`  | `<div class="header">`  |
| 导航     | `<nav>`     | `<div class="nav">`     |
| 主内容   | `<main>`    | `<div class="main">`    |
| 文章     | `<article>` | `<div class="article">` |
| 侧边栏   | `<aside>`   | `<div class="sidebar">` |
| 页脚     | `<footer>`  | `<div class="footer">`  |

## 🔗 相关资源

- [HTML 入门](/docs/frontend/html/)
- [可访问性](/docs/react/accessibility)
- [CSS 入门](/docs/frontend/css/)

---

**下一步**：学习 [CSS 入门](/docs/frontend/css/) 为网页添加样式。
