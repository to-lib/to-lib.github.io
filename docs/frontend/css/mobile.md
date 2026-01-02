---
sidebar_position: 7
title: 移动端适配
---

# 移动端适配

> [!TIP]
> 移动端适配是现代前端必备技能，让网页在各种设备上都有良好的体验。

## 🎯 Viewport 设置

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
```

| 属性            | 说明                     |
| --------------- | ------------------------ |
| `width`         | 视口宽度，`device-width` |
| `initial-scale` | 初始缩放比例             |
| `maximum-scale` | 最大缩放比例             |
| `user-scalable` | 是否允许用户缩放         |

## 📐 适配方案

### 1. rem 方案

根据根元素字体大小计算。

```javascript
// 动态设置根字体大小
function setRem() {
  const baseSize = 16;
  const designWidth = 375; // 设计稿宽度
  const scale = document.documentElement.clientWidth / designWidth;
  document.documentElement.style.fontSize = baseSize * scale + "px";
}

setRem();
window.addEventListener("resize", setRem);
```

```css
/* 使用 rem */
.title {
  font-size: 1.5rem; /* 24px at 375px */
  padding: 1rem;
}
```

### 2. vw/vh 方案（推荐）

直接使用视口单位，无需 JS。

```css
/* 设计稿 375px，元素 100px */
/* 100 / 375 * 100 = 26.67vw */

.box {
  width: 26.67vw;
  padding: 4vw;
  font-size: 4.27vw; /* 16px */
}
```

#### PostCSS 自动转换

```javascript
// postcss.config.js
module.exports = {
  plugins: {
    "postcss-px-to-viewport": {
      viewportWidth: 375,
      unitPrecision: 5,
      viewportUnit: "vw",
      minPixelValue: 1,
    },
  },
};
```

```css
/* 写 px，自动转 vw */
.box {
  width: 100px; /* → 26.67vw */
  font-size: 16px; /* → 4.27vw */
}
```

### 3. 响应式布局

结合媒体查询。

```css
.container {
  padding: 16px;
}

@media (min-width: 768px) {
  .container {
    max-width: 720px;
    margin: 0 auto;
  }
}

@media (min-width: 1024px) {
  .container {
    max-width: 960px;
  }
}
```

## 📱 1px 问题

高清屏上 1px 看起来很粗。

### 方案 1：伪元素 + transform

```css
.border-1px {
  position: relative;
}

.border-1px::after {
  content: "";
  position: absolute;
  left: 0;
  bottom: 0;
  width: 100%;
  height: 1px;
  background: #ccc;
  transform: scaleY(0.5);
  transform-origin: 0 0;
}

/* 兼容不同像素比 */
@media (-webkit-min-device-pixel-ratio: 3) {
  .border-1px::after {
    transform: scaleY(0.333);
  }
}
```

### 方案 2：box-shadow

```css
.border-1px {
  box-shadow: 0 1px 0 0 rgba(0, 0, 0, 0.1);
}
```

### 方案 3：svg border-image

```css
.border-1px {
  border-width: 1px;
  border-image: url("data:image/svg+xml,...") 2 stretch;
}
```

## 👆 触摸事件

### 点击延迟解决

```css
/* 移除 300ms 延迟 */
html {
  touch-action: manipulation;
}
```

### 触摸事件

```javascript
element.addEventListener("touchstart", (e) => {
  const touch = e.touches[0];
  console.log(touch.clientX, touch.clientY);
});

element.addEventListener("touchmove", (e) => {
  e.preventDefault(); // 阻止滚动
});

element.addEventListener("touchend", (e) => {
  console.log("触摸结束");
});
```

### 手势封装

```javascript
class Gesture {
  constructor(element) {
    this.element = element;
    this.startX = 0;
    this.startY = 0;

    element.addEventListener("touchstart", this.onStart.bind(this));
    element.addEventListener("touchend", this.onEnd.bind(this));
  }

  onStart(e) {
    this.startX = e.touches[0].clientX;
    this.startY = e.touches[0].clientY;
  }

  onEnd(e) {
    const endX = e.changedTouches[0].clientX;
    const endY = e.changedTouches[0].clientY;
    const deltaX = endX - this.startX;
    const deltaY = endY - this.startY;

    if (Math.abs(deltaX) > Math.abs(deltaY)) {
      if (deltaX > 50) this.onSwipeRight?.();
      if (deltaX < -50) this.onSwipeLeft?.();
    } else {
      if (deltaY > 50) this.onSwipeDown?.();
      if (deltaY < -50) this.onSwipeUp?.();
    }
  }
}
```

## 📦 安全区域

适配 iPhone 刘海屏。

```css
/* 底部安全区域 */
.footer {
  padding-bottom: env(safe-area-inset-bottom);
}

/* 全面适配 */
.container {
  padding-top: env(safe-area-inset-top);
  padding-right: env(safe-area-inset-right);
  padding-bottom: env(safe-area-inset-bottom);
  padding-left: env(safe-area-inset-left);
}
```

需要配合 viewport：

```html
<meta
  name="viewport"
  content="width=device-width, initial-scale=1.0, viewport-fit=cover"
/>
```

## 🎨 移动端优化

### 禁止选中

```css
.no-select {
  -webkit-user-select: none;
  user-select: none;
}
```

### 禁止长按菜单

```css
.no-callout {
  -webkit-touch-callout: none;
}
```

### 滚动优化

```css
.scroll-container {
  overflow-y: auto;
  -webkit-overflow-scrolling: touch; /* 惯性滚动 */
  overscroll-behavior: contain; /* 防止滚动穿透 */
}
```

### 输入框优化

```css
input,
textarea {
  /* 禁止自动放大 */
  font-size: 16px;
  /* 禁止自动大写 */
  text-transform: none;
}
```

```html
<!-- 调起数字键盘 -->
<input type="tel" pattern="[0-9]*" inputmode="numeric" />
```

## 📱 调试技巧

### Chrome 模拟器

1. 打开 DevTools (F12)
2. 点击设备图标或 Ctrl+Shift+M
3. 选择设备或自定义尺寸

### 真机调试

```bash
# Android
chrome://inspect

# iOS (需要 Mac)
# Safari → 开发 → 设备名
```

## 💡 最佳实践

1. **移动优先** - 先写移动端样式，再用媒体查询适配大屏
2. **使用 vw** - 推荐 vw 方案，简单无依赖
3. **触摸友好** - 点击区域至少 44px
4. **测试真机** - 模拟器不能完全代替真机测试

## 🔗 相关资源

- [CSS 响应式](/docs/frontend/css/responsive)
- [CSS 新特性](/docs/frontend/css/modern-css)

---

**下一步**：学习 [跨域详解](/docs/frontend/browser/cors) 掌握跨域解决方案。
