---
sidebar_position: 12
title: React Compiler
---

# React Compiler（自动性能优化）

> [!TIP]
> React Compiler 会在构建阶段自动重写你的组件与 Hooks 代码，减少不必要的重新渲染与手动 `memo`/`useMemo`/`useCallback` 的负担。

## 🎯 解决什么问题

在大型 React 应用中，“**状态变更导致的连锁渲染**”往往是性能瓶颈的来源。

传统优化手段包括：

- `React.memo`：避免子组件在 props 不变时重新渲染
- `useMemo`：缓存昂贵计算结果
- `useCallback`：缓存回调引用，减少子组件渲染

但这些手段有两个痛点：

- 需要开发者手动介入，且容易遗漏/误用
- 优化代码会增加复杂度，降低可读性

React Compiler 的目标是：**在不改变你写组件方式的前提下，尽可能自动完成 memoization**。

## ✅ 适用场景

- 组件树较深、频繁交互更新（列表、表格、编辑器、Dashboard）
- 多处使用 `memo`/`useMemo`/`useCallback` 进行手动调参
- 希望在不大改架构的前提下获得更稳定的性能表现

## 📦 安装

React Compiler 以 Babel 插件形式集成。

```bash
pnpm install -D babel-plugin-react-compiler@latest
```

> [!IMPORTANT] > **React Compiler 必须在 Babel 插件链中第一个运行**，否则可能无法正确分析源码。

## 🔧 基础配置

### Babel

```js
// babel.config.js
module.exports = {
  plugins: [
    "babel-plugin-react-compiler", // must run first!
    // ... other plugins
  ],
};
```

### Vite

如果你使用 `@vitejs/plugin-react`：

```js
// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: ["babel-plugin-react-compiler"],
      },
    }),
  ],
});
```

## 🧩 常用配置选项

多数 React 19 应用可以 **零配置** 运行。

当你需要更精细控制时，可以传入配置对象：

```js
// babel.config.js
module.exports = {
  plugins: [
    [
      "babel-plugin-react-compiler",
      {
        // 生产环境建议：遇到不符合 Rules of React 的代码时跳过而不是直接失败
        panicThreshold: "none",
      },
    ],
  ],
};
```

你可能会用到的选项：

- **`panicThreshold`**：遇到问题代码时是失败构建还是跳过
- **`target`**：目标 React 版本（17/18/19）
- **`compilationMode`**：选择编译策略（例如逐步启用）
- **`logger`**：输出哪些文件被编译
- **`gating`**：按运行时开关逐步灰度启用

## 🧭 渐进式启用建议

在老项目中建议采用“可回滚”的渐进策略：

- 先在一个业务模块内启用
- 或只对少量组件启用（基于 `compilationMode`）
- 保持 `panicThreshold: "none"`，避免阻塞 CI

### 使用 eslint 插件检查兼容性

```bash
pnpm install -D eslint-plugin-react-compiler
```

```js
// eslint.config.js
import reactCompiler from "eslint-plugin-react-compiler";

export default [
  {
    plugins: {
      "react-compiler": reactCompiler,
    },
    rules: {
      "react-compiler/react-compiler": "error",
    },
  },
];
```

### 跳过特定组件

使用 `"use no memo"` 指令跳过编译：

```jsx
function SpecialComponent() {
  "use no memo"; // 编译器会跳过此组件

  // 某些特殊逻辑...
  return <div>...</div>;
}
```

## 🔍 调试与验证

### 验证编译器是否生效

```jsx
// 开发模式下，编译器会在控制台输出信息
// 你也可以使用 React DevTools 的 Profiler 对比性能
```

### 查看编译结果

```js
// babel.config.js
module.exports = {
  plugins: [
    [
      "babel-plugin-react-compiler",
      {
        logger: {
          logEvent(filename, event) {
            console.log(`[Compiler] ${filename}:`, event);
          },
        },
      },
    ],
  ],
};
```

### Next.js 配置

```js
// next.config.js
module.exports = {
  experimental: {
    reactCompiler: true,
  },
};
```

## ❓ 常见问题

### 编译器会破坏我的代码吗？

编译器只会优化符合 **Rules of React** 的代码。如果你的代码违反了规则（如在渲染期间修改 state），编译器会跳过该组件。

### 我还需要 useMemo/useCallback 吗？

编译器启用后，大多数情况下**不再需要手动编写**这些优化代码。但保留现有代码也不会有问题。

### 对包体积有影响吗？

编译器在构建时运行，不会增加运行时体积。生成的代码可能略有变化，但通常可以忽略不计。

### 支持 TypeScript 吗？

完全支持。编译器在类型检查后的 AST 阶段工作。

## 📊 效果对比

| 场景                | 优化前     | 优化后   |
| ------------------- | ---------- | -------- |
| 列表滚动（1000 项） | ~16ms/帧   | ~4ms/帧  |
| 表单输入响应        | 明显卡顿   | 流畅     |
| 复杂 Dashboard      | 频繁重渲染 | 精准更新 |

## 💡 最佳实践

### 1. 遵循 Rules of React

```jsx
// ✅ 好：纯函数组件
function Good({ items }) {
  const filtered = items.filter((x) => x.active);
  return <List items={filtered} />;
}

// ❌ 坏：渲染期间有副作用
function Bad({ items }) {
  items.sort(); // 修改了输入！
  return <List items={items} />;
}
```

### 2. 先用 ESLint 检查

在启用编译器前，先用 `eslint-plugin-react-compiler` 扫描代码库，修复潜在问题。

### 3. 监控性能指标

使用 React DevTools Profiler 对比启用前后的渲染次数和时间。

## 🔗 相关资源

- [React Compiler 官方文档](https://react.dev/learn/react-compiler)
- [Rules of React](https://react.dev/reference/rules)
- [性能优化](/docs/react/performance-optimization)
- [并发渲染](/docs/react/concurrent-rendering)

---

**下一步**：了解 [并发渲染](/docs/react/concurrent-rendering) 进一步提升应用性能。
