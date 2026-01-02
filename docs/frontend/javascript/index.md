---
sidebar_position: 1
title: JavaScript 入门
---

# JavaScript 基础

> [!TIP]
> JavaScript 是网页的编程语言，让网页具有交互性和动态功能。

## 🎯 什么是 JavaScript？

JavaScript 是一种：

- **脚本语言** - 解释执行，无需编译
- **动态类型** - 变量类型可变
- **多范式** - 支持函数式、面向对象
- **事件驱动** - 响应用户交互

## 📦 引入方式

### 外部脚本（推荐）

```html
<script src="script.js"></script>

<!-- 或放在 body 末尾 -->
<body>
  <!-- 内容 -->
  <script src="script.js"></script>
</body>

<!-- defer: HTML 解析完后执行 -->
<script src="script.js" defer></script>

<!-- async: 下载后立即执行 -->
<script src="script.js" async></script>
```

### 内联脚本

```html
<script>
  console.log("Hello, World!");
</script>
```

## 🔧 基础语法

### 输出

```javascript
console.log("控制台输出");
alert("弹窗");
document.write("写入页面");
```

### 注释

```javascript
// 单行注释

/*
  多行注释
*/
```

### 语句与分号

```javascript
// 每行一个语句，分号可选但推荐
let name = "Alice";
console.log(name);
```

## 📝 变量

### 声明方式

```javascript
// const - 常量（推荐）
const PI = 3.14159;

// let - 可变变量（推荐）
let count = 0;
count = 1;

// var - 旧方式（避免使用）
var oldWay = "deprecated";
```

### 变量命名

```javascript
// 驼峰命名
let userName = "Alice";
let isActive = true;
const MAX_SIZE = 100;

// 有效名称
let _private = 1;
let $element = 2;
let camelCase = 3;

// 无效名称
// let 1name = 'error';
// let my-var = 'error';
```

## 🎨 数据类型

### 基本类型

```javascript
// 字符串
const str = "Hello";
const str2 = "World";
const template = `Hello, ${name}!`;

// 数字
const num = 42;
const float = 3.14;

// 布尔
const isTrue = true;
const isFalse = false;

// undefined
let notDefined;

// null
const empty = null;

// Symbol
const sym = Symbol("unique");

// BigInt
const big = 9007199254740991n;
```

### 引用类型

```javascript
// 对象
const person = {
  name: "Alice",
  age: 25,
};

// 数组
const numbers = [1, 2, 3, 4, 5];

// 函数
const greet = function (name) {
  return `Hello, ${name}`;
};
```

### 类型检测

```javascript
typeof "hello"; // 'string'
typeof 42; // 'number'
typeof true; // 'boolean'
typeof undefined; // 'undefined'
typeof null; // 'object' (历史遗留)
typeof {}; // 'object'
typeof []; // 'object'
typeof function () {}; // 'function'

Array.isArray([]); // true
```

## 🔢 运算符

### 算术运算符

```javascript
5 + 3; // 8
5 - 3; // 2
5 * 3; // 15
5 / 3; // 1.666...
5 % 3; // 2 (取余)
5 ** 3; // 125 (幂)

let a = 1;
a++; // 2
a--; // 1
```

### 比较运算符

```javascript
5 == "5"; // true (类型转换)
5 === "5"; // false (严格相等，推荐)
5 !== "5"; // true

5 > 3; // true
5 >= 5; // true
```

### 逻辑运算符

```javascript
true && false; // false
true || false; // true
!true; // false

// 短路求值
const name = userName || "Guest";
const safe = obj && obj.property;
```

## 🔗 相关资源

- [基础语法](/docs/frontend/javascript/fundamentals)
- [DOM 操作](/docs/frontend/javascript/dom)
- [异步编程](/docs/frontend/javascript/async)
- [ES6+](/docs/frontend/javascript/es6)

---

**下一步**：学习 [基础语法](/docs/frontend/javascript/fundamentals) 了解函数和对象。
