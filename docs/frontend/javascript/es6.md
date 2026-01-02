---
sidebar_position: 5
title: ES6+
---

# ES6+ 现代 JavaScript

> [!TIP]
> ES6（ECMAScript 2015）及后续版本带来了大量新特性，让 JavaScript 更强大、更易用。

## 🎯 变量声明

### let 和 const

```javascript
// const - 常量（推荐默认使用）
const PI = 3.14159;
const user = { name: "Alice" };
user.name = "Bob"; // ✅ 可以修改属性
// user = {};       // ❌ 不能重新赋值

// let - 可变变量
let count = 0;
count = 1; // ✅

// var - 避免使用（函数作用域，有变量提升）
```

### 块级作用域

```javascript
if (true) {
  let x = 1;
  const y = 2;
}
// x, y 在这里不可访问
```

## 📝 模板字符串

```javascript
const name = "Alice";
const age = 25;

// 模板字符串
const message = `Hello, ${name}! You are ${age} years old.`;

// 多行
const html = `
  <div class="card">
    <h2>${name}</h2>
  </div>
`;

// 表达式
const result = `Total: ${price * quantity}`;
```

## 🔧 解构赋值

### 数组解构

```javascript
const [a, b, c] = [1, 2, 3];

// 跳过元素
const [first, , third] = [1, 2, 3];

// 默认值
const [x = 0, y = 0] = [1];

// 剩余元素
const [head, ...tail] = [1, 2, 3, 4];
// head = 1, tail = [2, 3, 4]

// 交换变量
[a, b] = [b, a];
```

### 对象解构

```javascript
const user = { name: "Alice", age: 25, city: "Beijing" };

const { name, age } = user;

// 重命名
const { name: userName } = user;

// 默认值
const { country = "China" } = user;

// 嵌套
const {
  address: { street },
} = { address: { street: "Main" } };
```

## ⚡ 箭头函数

```javascript
// 基础
const add = (a, b) => a + b;

// 无参数
const greet = () => "Hello";

// 单参数（可省略括号）
const double = (n) => n * 2;

// 多行
const process = (data) => {
  const result = data.filter((x) => x > 0);
  return result;
};

// 返回对象（需要括号）
const createUser = (name) => ({ name, id: Date.now() });

// this 继承外层
const obj = {
  name: "Alice",
  greet() {
    setTimeout(() => {
      console.log(this.name); // 'Alice'
    }, 100);
  },
};
```

## 🔄 展开运算符

```javascript
// 数组展开
const arr1 = [1, 2, 3];
const arr2 = [...arr1, 4, 5]; // [1, 2, 3, 4, 5]

// 对象展开
const obj1 = { a: 1, b: 2 };
const obj2 = { ...obj1, c: 3 }; // { a: 1, b: 2, c: 3 }

// 合并对象（后面覆盖前面）
const merged = { ...defaults, ...userSettings };

// 函数参数
function sum(...numbers) {
  return numbers.reduce((a, b) => a + b, 0);
}
sum(1, 2, 3, 4); // 10
```

## 📦 对象简写

```javascript
const name = "Alice";
const age = 25;

// 属性简写
const user = { name, age }; // { name: 'Alice', age: 25 }

// 方法简写
const obj = {
  greet() {
    return "Hello";
  },
  async fetchData() {
    // ...
  },
};

// 计算属性名
const key = "dynamicKey";
const obj = {
  [key]: "value",
  [`prefix_${key}`]: "another",
};
```

## 🔀 可选链和空值合并

```javascript
// 可选链（?.）
const city = user?.address?.city; // 安全访问
const name = users?.[0]?.name; // 数组
const result = obj?.method?.(); // 方法

// 空值合并（??）
const value = null ?? "default"; // 'default'
const count = 0 ?? 10; // 0（只检查 null/undefined）

// 对比 ||
const count = 0 || 10; // 10（0 也是假值）
```

## 📋 数组新方法

```javascript
// 查找
arr.find((x) => x > 5); // 第一个匹配
arr.findIndex((x) => x > 5); // 第一个匹配的索引
arr.includes(5); // 是否包含

// 扁平化
[
  [1, 2],
  [3, 4],
].flat(); // [1, 2, 3, 4]
[1, [2, [3]]].flat(2); // [1, 2, 3]

// flatMap
[1, 2].flatMap((x) => [x, x * 2]); // [1, 2, 2, 4]

// at（支持负索引）
arr.at(-1); // 最后一个元素
```

## 📦 类

```javascript
class Person {
  // 私有字段
  #privateField = "secret";

  // 公共字段
  name;

  constructor(name) {
    this.name = name;
  }

  // 方法
  greet() {
    return `Hi, I'm ${this.name}`;
  }

  // 静态方法
  static create(name) {
    return new Person(name);
  }

  // getter/setter
  get info() {
    return `${this.name}`;
  }

  set info(value) {
    this.name = value;
  }
}

// 继承
class Student extends Person {
  constructor(name, grade) {
    super(name);
    this.grade = grade;
  }
}
```

## 📦 模块

### 导出

```javascript
// 命名导出
export const PI = 3.14159;
export function add(a, b) {
  return a + b;
}
export class User {}

// 默认导出
export default function main() {}

// 统一导出
const a = 1;
const b = 2;
export { a, b };
```

### 导入

```javascript
// 命名导入
import { PI, add } from "./math.js";

// 重命名
import { add as sum } from "./math.js";

// 默认导入
import main from "./main.js";

// 全部导入
import * as math from "./math.js";

// 动态导入
const module = await import("./module.js");
```

## 🔗 相关资源

- [JavaScript 入门](/docs/frontend/javascript/)
- [React 开发指南](/docs/react)
- [TypeScript](/docs/react/typescript)

---

**恭喜！** 你已完成前端基础学习。接下来可以学习 [React](/docs/react) 开始现代前端框架之旅！
