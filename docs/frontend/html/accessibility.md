---
sidebar_position: 6
title: 无障碍开发
---

# 无障碍开发 (Accessibility)

> [!TIP]
> 无障碍开发让所有人都能使用你的网站，包括残障人士。这也有助于 SEO 和整体用户体验。

## 🎯 为什么重要

- **包容性**：全球约 15% 人口有某种形式的残障
- **法律要求**：许多国家有无障碍法规
- **更好的 SEO**：搜索引擎也受益于语义化
- **更好的 UX**：对所有用户都更友好

## 📝 语义化 HTML

使用正确的 HTML 元素是无障碍的基础：

```html
<!-- ❌ 不好 -->
<div class="button" onclick="submit()">提交</div>
<div class="header">导航</div>

<!-- ✅ 好 -->
<button type="submit">提交</button>
<nav>导航</nav>
```

### 语义化标签

| 元素        | 用途           |
| ----------- | -------------- |
| `<header>`  | 页头           |
| `<nav>`     | 导航           |
| `<main>`    | 主内容（唯一） |
| `<article>` | 独立内容       |
| `<section>` | 内容分组       |
| `<aside>`   | 侧边栏         |
| `<footer>`  | 页脚           |

### 标题层级

```html
<h1>页面主标题</h1>
<h2>章节标题</h2>
<h3>子章节</h3>
<h2>另一章节</h2>
```

## 🏷️ ARIA 属性

ARIA（Accessible Rich Internet Applications）增强辅助技术支持。

### 常用角色

```html
<div role="button" tabindex="0">自定义按钮</div>
<div role="alert">重要通知</div>
<div role="dialog" aria-modal="true">弹窗</div>
<ul role="menu">
  <li role="menuitem">选项1</li>
</ul>
```

### 常用属性

```html
<!-- 标签 -->
<button aria-label="关闭">×</button>
<input aria-labelledby="label-id" />

<!-- 描述 -->
<button aria-describedby="hint">提交</button>
<p id="hint">点击后将发送邮件</p>

<!-- 状态 -->
<button aria-pressed="true">已选中</button>
<div aria-expanded="false">可展开内容</div>
<input aria-invalid="true" />
<div aria-busy="true">加载中</div>

<!-- 隐藏 -->
<div aria-hidden="true">仅视觉装饰</div>
```

### 实时区域

```html
<!-- 通知用户内容变化 -->
<div aria-live="polite">新消息将在这里显示</div>
<div aria-live="assertive">紧急！立即通知</div>
```

## ⌨️ 键盘导航

确保所有交互可通过键盘完成：

### Tab 顺序

```html
<!-- 自然顺序 -->
<input tabindex="0" />
<!-- 正常 Tab 顺序 -->
<button tabindex="-1">跳过</button>
<!-- 不可 Tab，但可聚焦 -->

<!-- 避免使用正数 tabindex -->
```

### 焦点可见

```css
/* 不要移除焦点样式 */
:focus {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
}

/* 只在键盘导航时显示 */
:focus:not(:focus-visible) {
  outline: none;
}

:focus-visible {
  outline: 2px solid #3b82f6;
}
```

### 键盘事件

```javascript
element.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    handleClick();
  }

  // 方向键导航
  if (e.key === "ArrowDown") {
    focusNext();
  }

  // ESC 关闭
  if (e.key === "Escape") {
    closeModal();
  }
});
```

### 焦点陷阱

```javascript
// 弹窗内焦点循环
function trapFocus(container) {
  const focusable = container.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  const first = focusable[0];
  const last = focusable[focusable.length - 1];

  container.addEventListener("keydown", (e) => {
    if (e.key === "Tab") {
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  });
}
```

## 🖼️ 图像无障碍

### Alt 文本

```html
<!-- 信息性图片 -->
<img src="chart.png" alt="2024年销售增长20%的柱状图" />

<!-- 装饰性图片 -->
<img src="decoration.png" alt="" />

<!-- 复杂图片 -->
<figure>
  <img src="diagram.png" alt="系统架构图" />
  <figcaption>图1：系统由前端、API 网关和微服务组成...</figcaption>
</figure>
```

### 图标

```html
<!-- 带文字的图标 -->
<button>
  <svg aria-hidden="true">...</svg>
  保存
</button>

<!-- 仅图标 -->
<button aria-label="保存">
  <svg aria-hidden="true">...</svg>
</button>
```

## 📋 表单无障碍

```html
<form>
  <!-- 关联标签 -->
  <label for="email">邮箱</label>
  <input
    type="email"
    id="email"
    aria-describedby="email-hint"
    aria-required="true"
  />
  <p id="email-hint">请输入有效的邮箱地址</p>

  <!-- 必填提示 -->
  <label for="name">
    姓名 <span aria-hidden="true">*</span>
    <span class="sr-only">（必填）</span>
  </label>
  <input type="text" id="name" required />

  <!-- 错误状态 -->
  <input type="text" aria-invalid="true" aria-describedby="error-msg" />
  <p id="error-msg" role="alert">请输入有效值</p>
</form>
```

### 屏幕阅读器专用文本

```css
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
```

## 🎨 颜色与对比度

### 对比度要求

- **普通文本**：至少 4.5:1
- **大文本（18px+粗体或 24px+）**：至少 3:1
- **UI 组件**：至少 3:1

```css
/* ✅ 好对比度 */
.text {
  color: #374151; /* 灰 */
  background: #fff;
}

/* ❌ 对比度不足 */
.low-contrast {
  color: #9ca3af;
  background: #f3f4f6;
}
```

### 不仅依赖颜色

```html
<!-- ❌ 仅用颜色区分 -->
<span style="color: red;">错误</span>
<span style="color: green;">成功</span>

<!-- ✅ 添加图标或文字 -->
<span style="color: red;">❌ 错误</span>
<span style="color: green;">✓ 成功</span>
```

## 🔧 测试工具

- **浏览器扩展**：axe DevTools, WAVE
- **键盘测试**：不用鼠标操作网站
- **屏幕阅读器**：VoiceOver (Mac), NVDA (Windows)
- **对比度检查**：WebAIM Contrast Checker

## 🔗 相关资源

- [语义化 HTML](/docs/frontend/html/semantic)
- [表单](/docs/frontend/html/forms)

---

**下一步**：学习 [前端性能优化](/docs/frontend/advanced/performance) 提升加载速度。
