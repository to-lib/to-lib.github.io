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

> [!IMPORTANT]
> **React Compiler 必须在 Babel 插件链中第一个运行**，否则可能无法正确分析源码。

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

## 🔗 相关资源

- [React Compiler Installation（官方）](https://react.dev/learn/react-compiler/installation)
- [React Compiler Configuration（官方）](https://react.dev/reference/react-compiler/configuration)
- [性能优化](/docs/react/performance-optimization)
