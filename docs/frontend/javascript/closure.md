---
sidebar_position: 6
title: 闭包与作用域
---

# 闭包与作用域

> [!TIP]
> 闭包是 JavaScript 中最重要的概念之一，理解它有助于写出更优雅的代码。

## 🎯 作用域

作用域决定了变量的可访问范围。

### 全局作用域

```javascript
const globalVar = "我是全局变量";

function test() {
  console.log(globalVar); // 可以访问
}
```

### 函数作用域

```javascript
function outer() {
  const localVar = "我是局部变量";
  console.log(localVar); // ✅ 可以访问
}

console.log(localVar); // ❌ ReferenceError
```

### 块级作用域 (ES6+)

```javascript
if (true) {
  let blockVar = "let 有块级作用域";
  const blockConst = "const 也是";
  var noBlockScope = "var 没有";
}

console.log(noBlockScope); // ✅ 可以访问
console.log(blockVar); // ❌ ReferenceError
```

## 🔍 词法作用域

JavaScript 使用词法作用域（静态作用域），函数的作用域在**定义时**确定：

```javascript
const name = "Global";

function outer() {
  const name = "Outer";

  function inner() {
    console.log(name); // 'Outer'，而非 'Global'
  }

  return inner;
}

const fn = outer();
fn(); // 'Outer'
```

## 📦 闭包

闭包 = **函数** + **其词法环境**

```javascript
function createCounter() {
  let count = 0; // 私有变量

  return function () {
    count++;
    return count;
  };
}

const counter = createCounter();
console.log(counter()); // 1
console.log(counter()); // 2
console.log(counter()); // 3
```

### 闭包的特点

1. **函数可以访问定义时的外部变量**
2. **外部变量在函数调用后仍然存活**
3. **每次调用外层函数创建新的闭包**

```javascript
const counter1 = createCounter();
const counter2 = createCounter();

console.log(counter1()); // 1
console.log(counter1()); // 2
console.log(counter2()); // 1 (独立的闭包)
```

## 💡 常见应用

### 1. 私有变量

```javascript
function createPerson(name) {
  let _age = 0; // 私有

  return {
    getName: () => name,
    getAge: () => _age,
    setAge: (age) => {
      if (age > 0) _age = age;
    },
  };
}

const person = createPerson("Alice");
person.setAge(25);
console.log(person.getAge()); // 25
console.log(person._age); // undefined (无法直接访问)
```

### 2. 函数工厂

```javascript
function multiply(x) {
  return function (y) {
    return x * y;
  };
}

const double = multiply(2);
const triple = multiply(3);

console.log(double(5)); // 10
console.log(triple(5)); // 15
```

### 3. 模块模式

```javascript
const Calculator = (function () {
  let result = 0;

  return {
    add: (x) => (result += x),
    subtract: (x) => (result -= x),
    getResult: () => result,
    reset: () => (result = 0),
  };
})();

Calculator.add(10);
Calculator.subtract(3);
console.log(Calculator.getResult()); // 7
```

### 4. 事件处理器

```javascript
function setupButtons() {
  for (let i = 1; i <= 3; i++) {
    document.getElementById(`btn${i}`).onclick = function () {
      console.log(`Button ${i} clicked`);
    };
  }
}
```

> [!WARNING]
> 使用 `var` 时会有经典的循环闭包问题，使用 `let` 可以避免。

## ⚠️ 常见陷阱

### 循环中的闭包问题

```javascript
// ❌ 错误：var 没有块级作用域
for (var i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 100);
}
// 输出: 3, 3, 3

// ✅ 方案1：使用 let
for (let i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 100);
}
// 输出: 0, 1, 2

// ✅ 方案2：使用 IIFE
for (var i = 0; i < 3; i++) {
  ((j) => {
    setTimeout(() => console.log(j), 100);
  })(i);
}
```

### 内存泄漏

```javascript
// ❌ 可能导致内存泄漏
function attachHandler() {
  const largeData = new Array(1000000);

  element.onclick = function () {
    // largeData 会一直被保留在内存中
    console.log("clicked");
  };
}

// ✅ 只保留必要的数据
function attachHandler() {
  const largeData = new Array(1000000);
  const neededData = largeData.length;

  element.onclick = function () {
    console.log(neededData);
  };
}
```

## 🔗 相关资源

- [基础语法](/docs/frontend/javascript/fundamentals)
- [原型链](/docs/frontend/javascript/prototype)
- [ES6+](/docs/frontend/javascript/es6)

---

**下一步**：学习 [原型链与继承](/docs/frontend/javascript/prototype) 理解对象系统。
