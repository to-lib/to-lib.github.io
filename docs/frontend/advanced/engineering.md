---
sidebar_position: 3
title: 前端工程化
---

# 前端工程化

> [!TIP]
> 前端工程化让团队协作更高效，项目更易于维护和扩展。

## 📦 包管理器

### npm

```bash
# 初始化项目
npm init -y

# 安装依赖
npm install lodash
npm install -D typescript  # 开发依赖

# 常用命令
npm run dev
npm run build
npm update
npm outdated  # 检查过期依赖
```

### pnpm (推荐)

```bash
# 安装 pnpm
npm install -g pnpm

# 使用方式与 npm 类似
pnpm install
pnpm add lodash
pnpm add -D typescript
```

#### pnpm 优势

- **更快**：依赖只下载一次，硬链接复用
- **更省空间**：共享依赖存储
- **更严格**：避免幽灵依赖

## 🔧 构建工具

### Vite (推荐)

现代前端构建工具，开发体验极佳。

```bash
# 创建项目
pnpm create vite my-app --template react-ts

# 目录结构
my-app/
├── public/
├── src/
│   ├── App.tsx
│   └── main.tsx
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": "http://localhost:8080",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
```

### Webpack

功能强大，生态丰富。

```javascript
// webpack.config.js
const path = require("path");

module.exports = {
  entry: "./src/index.js",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "[name].[contenthash].js",
  },
  module: {
    rules: [
      {
        test: /\.jsx?$/,
        use: "babel-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"],
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: "./public/index.html",
    }),
  ],
};
```

## 📝 代码规范

### ESLint

```bash
# 安装
pnpm add -D eslint @eslint/js

# 初始化配置
npx eslint --init
```

```javascript
// eslint.config.js (Flat Config)
import js from "@eslint/js";

export default [
  js.configs.recommended,
  {
    rules: {
      "no-unused-vars": "warn",
      "no-console": "warn",
    },
  },
];
```

### Prettier

```bash
pnpm add -D prettier
```

```json
// .prettierrc
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5"
}
```

### EditorConfig

```ini
# .editorconfig
root = true

[*]
indent_style = space
indent_size = 2
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true
```

## 🔀 Git 工作流

### 提交规范

```bash
# Conventional Commits
feat: 新功能
fix: 修复 Bug
docs: 文档更新
style: 代码格式（不影响功能）
refactor: 重构
perf: 性能优化
test: 测试
chore: 构建/工具变动

# 示例
feat(auth): add login functionality
fix(ui): resolve button alignment issue
```

### Husky + lint-staged

```bash
# 安装
pnpm add -D husky lint-staged

# 初始化
npx husky init
```

```json
// package.json
{
  "lint-staged": {
    "*.{js,ts,tsx}": ["eslint --fix", "prettier --write"],
    "*.{css,md}": ["prettier --write"]
  }
}
```

## 🧪 测试

### 单元测试 (Vitest)

```bash
pnpm add -D vitest
```

```javascript
// sum.test.js
import { describe, it, expect } from "vitest";
import { sum } from "./sum";

describe("sum", () => {
  it("adds 1 + 2 to equal 3", () => {
    expect(sum(1, 2)).toBe(3);
  });
});
```

### 组件测试

```javascript
import { render, screen } from "@testing-library/react";
import Button from "./Button";

test("renders button with text", () => {
  render(<Button>Click me</Button>);
  expect(screen.getByText("Click me")).toBeInTheDocument();
});
```

## 🐛 调试技巧

### Chrome DevTools

```javascript
// 断点调试
debugger;

// 条件断点
// 右键代码行 -> Add conditional breakpoint

// 日志点（不暂停）
// 右键代码行 -> Add logpoint
```

### console 方法

```javascript
console.log("普通日志");
console.warn("警告");
console.error("错误");
console.table([{ a: 1 }, { a: 2 }]); // 表格形式
console.group("分组");
console.log("内容");
console.groupEnd();
console.time("计时");
// ... 代码
console.timeEnd("计时");
```

### Source Maps

```javascript
// vite.config.ts
export default defineConfig({
  build: {
    sourcemap: true, // 生成 source maps
  },
});
```

## 📁 项目结构

```
src/
├── assets/          # 静态资源
├── components/      # 通用组件
│   ├── Button/
│   │   ├── index.tsx
│   │   ├── Button.tsx
│   │   └── Button.css
│   └── index.ts     # 统一导出
├── hooks/           # 自定义 hooks
├── pages/           # 页面组件
├── services/        # API 请求
├── store/           # 状态管理
├── utils/           # 工具函数
├── types/           # TypeScript 类型
├── App.tsx
└── main.tsx
```

## 🔗 相关资源

- [ES6 模块化](/docs/frontend/javascript/modules)
- [前端性能优化](/docs/frontend/advanced/performance)

---

**恭喜**：你已完成前端基础学习！继续探索 [React](/docs/react) 构建现代应用。
