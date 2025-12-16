---
sidebar_position: 26
title: 可访问性（a11y）
---

# 可访问性（a11y）最佳实践

> [!TIP]
> 可访问性（Accessibility / a11y）让更多人能顺利使用你的产品：键盘用户、读屏用户、色弱用户、低网速/低性能设备用户等。

## ✅ 基本原则

- **优先使用语义化 HTML**：让浏览器与辅助技术理解你的页面
- **保证键盘可用**：Tab/Enter/Esc 等路径完整
- **可见的焦点（focus）**：用户能看见当前操作位置
- **合理的 ARIA**：只在必要时使用，用对比用多更重要

## 🧱 语义化优先（React 不会帮你补语义）

- 使用 `button` 而不是 `div` + onClick
- 使用 `label` 关联表单控件
- 使用正确的标题层级 `h1`~`h6`

```jsx
// ✗ 不推荐
<div onClick={onSave}>Save</div>

// ✅ 推荐
<button type="button" onClick={onSave}>
  Save
</button>
```

## ⌨️ 键盘支持与焦点管理

### 1) 不要移除 outline

很多 UI 会把 `outline: none;` 当“美化”，但这会让键盘用户迷失。

### 2) 弹窗/抽屉要管理焦点

- 打开时把焦点放到弹窗内的第一个可操作元素
- 关闭时把焦点还回触发按钮
- 支持 `Esc` 关闭

> 如果你使用 Headless UI / Radix UI / shadcn/ui 等组件库，通常已经内置了这些行为。

## 🏷️ ARIA 的常见用法

### aria-label

当按钮只有图标时，给出可读文本：

```jsx
<button aria-label="Close" onClick={onClose}>
  <IconX />
</button>
```

### aria-expanded / aria-controls

用于折叠菜单/下拉：

```jsx
<button
  aria-expanded={open}
  aria-controls="menu"
  onClick={() => setOpen((v) => !v)}
>
  Menu
</button>
<div id="menu" hidden={!open}>
  ...
</div>
```

## 🖼️ 图片与媒体

- 信息性图片必须提供 `alt`
- 装饰性图片可以用空 `alt=""`

```jsx
<img src="/logo.png" alt="to-lib logo" />
<img src="/bg.png" alt="" />
```

视频建议：

- 提供字幕/文字稿
- 避免自动播放（尤其带声音）

## 🎨 颜色与动效

- 文本对比度要足够（浅色文字 + 浅色背景是高频问题）
- 对动效敏感的用户应支持 `prefers-reduced-motion`

```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

## 🧪 如何做 a11y 测试

- **键盘走一遍主流程**：Tab/Shift+Tab/Enter/Esc
- **打开浏览器无障碍检查**（如 Lighthouse）
- **读屏测试（可选）**：macOS VoiceOver

## ✅ Checklist（上线前快速自查）

- 页面上所有可点击元素都能 Tab 到
- 焦点可见（focus ring 不被隐藏）
- 表单输入有 label / aria-label
- 图标按钮有可读名称
- Modal/Popover 支持 Esc 关闭
