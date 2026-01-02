---
sidebar_position: 9
title: 模块化
---

# JavaScript 模块化

> [!TIP]
> 模块化让代码更易于组织、复用和维护，是现代 JavaScript 开发的基础。

## 🎯 ES Modules (ESM)

ES Modules 是 JavaScript 官方的模块系统，现代浏览器和 Node.js 都支持。

### 导出 (export)

```javascript
// math.js

// 命名导出
export const PI = 3.14159;

export function add(a, b) {
  return a + b;
}

export function subtract(a, b) {
  return a - b;
}

// 也可以统一导出
const multiply = (a, b) => a * b;
const divide = (a, b) => a / b;

export { multiply, divide };
```

### 默认导出

```javascript
// logger.js

// 每个模块只能有一个默认导出
export default class Logger {
  log(message) {
    console.log(`[LOG] ${message}`);
  }

  error(message) {
    console.error(`[ERROR] ${message}`);
  }
}
```

### 导入 (import)

```javascript
// main.js

// 导入命名导出
import { PI, add, subtract } from "./math.js";

// 导入默认导出
import Logger from "./logger.js";

// 重命名导入
import { add as sum } from "./math.js";

// 导入全部
import * as MathUtils from "./math.js";
console.log(MathUtils.PI);
console.log(MathUtils.add(1, 2));

// 混合导入
import Logger, { PI, add } from "./combined.js";
```

### 重新导出

```javascript
// index.js - 统一导出多个模块

export { add, subtract } from "./math.js";
export { default as Logger } from "./logger.js";
export * from "./utils.js";
```

## 🔄 动态导入

动态导入用于按需加载模块，返回 Promise：

```javascript
// 按需加载
async function loadChart() {
  const { Chart } = await import("./chart.js");
  return new Chart();
}

// 条件加载
async function loadLocale(lang) {
  const locale = await import(`./locales/${lang}.js`);
  return locale.default;
}

// 配合 React.lazy
const LazyComponent = React.lazy(() => import("./HeavyComponent"));
```

## 📦 CommonJS (CJS)

Node.js 传统的模块系统：

```javascript
// math.js
const PI = 3.14159;

function add(a, b) {
  return a + b;
}

module.exports = { PI, add };
// 或
exports.PI = PI;
exports.add = add;
```

```javascript
// main.js
const { PI, add } = require("./math.js");
const math = require("./math.js");

console.log(PI);
console.log(math.add(1, 2));
```

## 🔀 ESM vs CommonJS

| 特性         | ES Modules     | CommonJS         |
| ------------ | -------------- | ---------------- |
| 加载时机     | 编译时（静态） | 运行时（动态）   |
| 导出         | `export`       | `module.exports` |
| 导入         | `import`       | `require()`      |
| 顶层 this    | `undefined`    | `module` 对象    |
| 浏览器支持   | ✅ 原生支持    | ❌ 需打包        |
| Tree Shaking | ✅ 支持        | ❌ 不支持        |

### 互操作

```javascript
// 在 ESM 中使用 CJS
import pkg from "cjs-package";
import { createRequire } from "module";
const require = createRequire(import.meta.url);

// 在 CJS 中使用 ESM
const esmModule = await import("esm-package");
```

## 📁 模块组织

### 目录结构

```
src/
├── components/
│   ├── Button/
│   │   ├── index.js
│   │   └── Button.css
│   └── index.js      # 统一导出
├── utils/
│   ├── format.js
│   ├── validate.js
│   └── index.js
└── index.js          # 应用入口
```

### Barrel 导出模式

```javascript
// components/index.js
export { Button } from "./Button";
export { Input } from "./Input";
export { Modal } from "./Modal";

// 使用时
import { Button, Input, Modal } from "./components";
```

## 🌐 浏览器中使用

```html
<!-- 使用 type="module" -->
<script type="module">
  import { greet } from "./greet.js";
  greet("World");
</script>

<!-- 外部模块 -->
<script type="module" src="./main.js"></script>

<!-- 兼容不支持模块的浏览器 -->
<script nomodule src="./fallback.js"></script>
```

### 导入映射 (Import Maps)

```html
<script type="importmap">
  {
    "imports": {
      "lodash": "https://cdn.skypack.dev/lodash",
      "@/utils": "./src/utils/index.js"
    }
  }
</script>

<script type="module">
  import _ from "lodash";
  import { format } from "@/utils";
</script>
```

## 💡 最佳实践

### 1. 优先使用 ESM

```javascript
// ✅ 推荐 - ESM
import { useState } from "react";

// ❌ 避免 - CommonJS（除非必要）
const { useState } = require("react");
```

### 2. 一个文件一个职责

```javascript
// ✅ 好 - 职责单一
// formatDate.js
export function formatDate(date) {
  /* ... */
}

// ❌ 不好 - 混杂多个功能
// utils.js
export function formatDate() {
  /* ... */
}
export function validateEmail() {
  /* ... */
}
export function fetchData() {
  /* ... */
}
```

### 3. 使用 index.js 简化导入

```javascript
// Button/index.js
export { default } from "./Button";
export * from "./types";

// 导入更简洁
import Button from "./Button"; // 自动找 index.js
```

### 4. 避免循环依赖

```javascript
// ❌ 避免
// a.js
import { b } from "./b.js";
export const a = "A";

// b.js
import { a } from "./a.js"; // 循环依赖！
export const b = "B";
```

## 🔗 相关资源

- [ES6+](/docs/frontend/javascript/es6)
- [异步编程](/docs/frontend/javascript/async)

---

**下一步**：学习 [正则表达式](/docs/frontend/javascript/regex) 处理文本匹配。
