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

### aria-live 动态内容通知

```jsx
function Notification({ message }) {
  return (
    <div aria-live="polite" aria-atomic="true">
      {message}
    </div>
  );
}

// 用于表单验证
function FormError({ error }) {
  return (
    <div role="alert" aria-live="assertive">
      {error}
    </div>
  );
}
```

### aria-describedby 补充说明

```jsx
function PasswordInput() {
  const id = useId();

  return (
    <div>
      <label htmlFor={`${id}-password`}>密码</label>
      <input
        id={`${id}-password`}
        type="password"
        aria-describedby={`${id}-hint`}
      />
      <p id={`${id}-hint`}>密码至少包含 8 个字符</p>
    </div>
  );
}
```

### 常用 ARIA 角色

| 角色                          | 用途                   |
| ----------------------------- | ---------------------- |
| `role="button"`               | 非 button 元素作为按钮 |
| `role="dialog"`               | 模态框                 |
| `role="alert"`                | 紧急通知               |
| `role="navigation"`           | 导航区域               |
| `role="main"`                 | 主要内容区             |
| `role="tablist/tab/tabpanel"` | 选项卡组件             |

## ⌨️ 键盘导航完整实现

### Tab 顺序管理

```jsx
// 使用 tabIndex 控制焦点顺序
function Card({ children }) {
  return (
    <div tabIndex={0}>
      {" "}
      {/* 可被 Tab 聚焦 */}
      {children}
    </div>
  );
}

// tabIndex 值说明
// 0: 按 DOM 顺序可聚焦
// -1: 可编程聚焦但不能 Tab 到达
// >0: 按数字顺序聚焦（不推荐）
```

### 键盘事件处理

```jsx
function InteractiveCard({ onClick }) {
  const handleKeyDown = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onClick();
    }
  };

  return (
    <div role="button" tabIndex={0} onClick={onClick} onKeyDown={handleKeyDown}>
      Click or press Enter
    </div>
  );
}
```

### 焦点陷阱（Focus Trap）

模态框需要将焦点限制在内部：

```jsx
function Modal({ isOpen, onClose, children }) {
  const modalRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;

    const modal = modalRef.current;
    const focusableElements = modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    const handleTab = (e) => {
      if (e.key !== "Tab") return;

      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    };

    modal.addEventListener("keydown", handleTab);
    firstElement?.focus();

    return () => modal.removeEventListener("keydown", handleTab);
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div ref={modalRef} role="dialog" aria-modal="true">
      {children}
      <button onClick={onClose}>关闭</button>
    </div>
  );
}
```

### 焦点恢复

```jsx
function useModalFocus(isOpen) {
  const previousFocus = useRef(null);

  useEffect(() => {
    if (isOpen) {
      previousFocus.current = document.activeElement;
    } else if (previousFocus.current) {
      previousFocus.current.focus();
    }
  }, [isOpen]);
}
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

### React 中检测用户偏好

```jsx
function useReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefersReducedMotion(mediaQuery.matches);

    const handler = (e) => setPrefersReducedMotion(e.matches);
    mediaQuery.addEventListener("change", handler);
    return () => mediaQuery.removeEventListener("change", handler);
  }, []);

  return prefersReducedMotion;
}

// 使用
function AnimatedComponent() {
  const prefersReducedMotion = useReducedMotion();

  return (
    <motion.div
      animate={{ x: 100 }}
      transition={{ duration: prefersReducedMotion ? 0 : 0.3 }}
    />
  );
}
```

## 🧪 如何做 a11y 测试

- **键盘走一遍主流程**：Tab/Shift+Tab/Enter/Esc
- **打开浏览器无障碍检查**（如 Lighthouse）
- **读屏测试（可选）**：macOS VoiceOver

### 自动化测试

```jsx
// 使用 @testing-library/jest-dom
import { render, screen } from "@testing-library/react";

test("button is accessible", () => {
  render(<Button>Click me</Button>);

  const button = screen.getByRole("button", { name: /click me/i });
  expect(button).toBeInTheDocument();
});

// 使用 axe-core
import { axe, toHaveNoViolations } from "jest-axe";
expect.extend(toHaveNoViolations);

test("has no a11y violations", async () => {
  const { container } = render(<App />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

## ✅ Checklist（上线前快速自查）

- [ ] 页面上所有可点击元素都能 Tab 到
- [ ] 焦点可见（focus ring 不被隐藏）
- [ ] 表单输入有 label / aria-label
- [ ] 图标按钮有可读名称
- [ ] Modal/Popover 支持 Esc 关闭
- [ ] 颜色对比度达到 WCAG AA 标准
- [ ] 动态内容有 aria-live 通知
- [ ] 可以仅用键盘完成主要流程

## 🔗 相关资源

- [Portals](/docs/react/portals) - 模态框实现
- [表单处理](/docs/react/forms) - 可访问表单
- [WCAG 指南](https://www.w3.org/WAI/WCAG21/quickref/)

---

**下一步**：使用 [Portals](/docs/react/portals) 实现无障碍的模态框组件。
