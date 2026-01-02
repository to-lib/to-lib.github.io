---
sidebar_position: 2
title: 常用元素
---

# HTML 常用元素

> [!TIP]
> 掌握常用 HTML 元素是构建网页的基础。本文介绍文本、链接、图片、列表、表格等核心元素。

## 📝 文本元素

### 标题

```html
<h1>一级标题（每页只用一次）</h1>
<h2>二级标题</h2>
<h3>三级标题</h3>
<h4>四级标题</h4>
<h5>五级标题</h5>
<h6>六级标题</h6>
```

### 段落和文本

```html
<p>这是一个段落。段落之间自动有间距。</p>

<span>行内文本</span>
<br />
<!-- 换行 -->
<hr />
<!-- 水平分隔线 -->
```

### 文本格式化

```html
<strong>重要文本（加粗）</strong>
<em>强调文本（斜体）</em>
<mark>高亮文本</mark>
<del>删除线</del>
<ins>下划线（插入文本）</ins>
<sub>下标</sub>
<sup>上标</sup>
<code>代码文本</code>
<pre>预格式化文本（保留空格和换行）</pre>
```

### 引用

```html
<!-- 块引用 -->
<blockquote>
  这是一段引用文本。
  <cite>—— 引用来源</cite>
</blockquote>

<!-- 行内引用 -->
<q>这是行内引用</q>
```

## 🔗 链接

### 基础链接

```html
<!-- 普通链接 -->
<a href="https://example.com">访问网站</a>

<!-- 新标签页打开 -->
<a href="https://example.com" target="_blank" rel="noopener">新标签页打开</a>

<!-- 页内锚点 -->
<a href="#section1">跳转到章节1</a>
<h2 id="section1">章节1</h2>

<!-- 邮件链接 -->
<a href="mailto:hello@example.com">发送邮件</a>

<!-- 电话链接 -->
<a href="tel:+8612345678900">拨打电话</a>

<!-- 下载链接 -->
<a href="file.pdf" download>下载文件</a>
```

### 链接属性

| 属性              | 作用                       |
| ----------------- | -------------------------- |
| `href`            | 链接目标地址               |
| `target="_blank"` | 新标签页打开               |
| `rel="noopener"`  | 安全属性（新标签页时使用） |
| `download`        | 下载而非打开               |
| `title`           | 鼠标悬停提示               |

## 🖼️ 图片

### 基础图片

```html
<img src="image.jpg" alt="图片描述" />

<!-- 指定尺寸 -->
<img src="image.jpg" alt="描述" width="300" height="200" />

<!-- 响应式图片 -->
<img src="image.jpg" alt="描述" style="max-width: 100%; height: auto;" />
```

### 响应式图片

```html
<!-- 不同分辨率 -->
<img
  src="image.jpg"
  srcset="image-320w.jpg 320w, image-640w.jpg 640w, image-1280w.jpg 1280w"
  sizes="(max-width: 320px) 280px,
         (max-width: 640px) 580px,
         1200px"
  alt="响应式图片"
/>

<!-- 不同设备 -->
<picture>
  <source media="(min-width: 800px)" srcset="large.jpg" />
  <source media="(min-width: 400px)" srcset="medium.jpg" />
  <img src="small.jpg" alt="图片描述" />
</picture>
```

### 图片与说明

```html
<figure>
  <img src="chart.png" alt="销售图表" />
  <figcaption>图1：2024年销售数据</figcaption>
</figure>
```

## 📋 列表

### 无序列表

```html
<ul>
  <li>苹果</li>
  <li>香蕉</li>
  <li>橙子</li>
</ul>
```

### 有序列表

```html
<ol>
  <li>第一步：准备材料</li>
  <li>第二步：开始制作</li>
  <li>第三步：完成</li>
</ol>

<!-- 自定义起始值 -->
<ol start="5">
  <li>第五项</li>
  <li>第六项</li>
</ol>
```

### 定义列表

```html
<dl>
  <dt>HTML</dt>
  <dd>超文本标记语言</dd>

  <dt>CSS</dt>
  <dd>层叠样式表</dd>
</dl>
```

### 嵌套列表

```html
<ul>
  <li>
    前端
    <ul>
      <li>HTML</li>
      <li>CSS</li>
      <li>JavaScript</li>
    </ul>
  </li>
  <li>
    后端
    <ul>
      <li>Java</li>
      <li>Python</li>
    </ul>
  </li>
</ul>
```

## 📊 表格

### 基础表格

```html
<table>
  <thead>
    <tr>
      <th>姓名</th>
      <th>年龄</th>
      <th>城市</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>张三</td>
      <td>25</td>
      <td>北京</td>
    </tr>
    <tr>
      <td>李四</td>
      <td>30</td>
      <td>上海</td>
    </tr>
  </tbody>
</table>
```

### 表格进阶

```html
<table>
  <caption>
    员工信息表
  </caption>
  <thead>
    <tr>
      <th scope="col">姓名</th>
      <th scope="col">部门</th>
      <th scope="col">工资</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>张三</td>
      <td rowspan="2">技术部</td>
      <!-- 合并行 -->
      <td>10000</td>
    </tr>
    <tr>
      <td>李四</td>
      <td>12000</td>
    </tr>
    <tr>
      <td colspan="2">合计</td>
      <!-- 合并列 -->
      <td>22000</td>
    </tr>
  </tbody>
</table>
```

## 📦 容器元素

### 块级容器

```html
<div>块级容器，独占一行</div>
```

### 行内容器

```html
<span>行内容器，不换行</span>
```

### 分组

```html
<div class="card">
  <h2>标题</h2>
  <p>内容...</p>
</div>
```

## 🎬 多媒体

### 视频

```html
<video width="640" height="360" controls>
  <source src="video.mp4" type="video/mp4" />
  <source src="video.webm" type="video/webm" />
  您的浏览器不支持视频播放。
</video>

<!-- 自动播放（静音） -->
<video autoplay muted loop playsinline>
  <source src="background.mp4" type="video/mp4" />
</video>
```

### 音频

```html
<audio controls>
  <source src="audio.mp3" type="audio/mpeg" />
  <source src="audio.ogg" type="audio/ogg" />
  您的浏览器不支持音频播放。
</audio>
```

### 嵌入内容

```html
<!-- iframe 嵌入 -->
<iframe
  src="https://www.youtube.com/embed/VIDEO_ID"
  width="560"
  height="315"
  frameborder="0"
  allowfullscreen
></iframe>
```

## 🔗 相关资源

- [HTML 入门](/docs/frontend/html/)
- [表单](/docs/frontend/html/forms)
- [语义化](/docs/frontend/html/semantic)

---

**下一步**：学习 [表单](/docs/frontend/html/forms) 创建用户输入界面。
