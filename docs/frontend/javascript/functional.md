---
sidebar_position: 14
title: 函数式编程
---

# JavaScript 函数式编程

> [!TIP]
> 函数式编程是一种编程范式，强调使用纯函数和不可变数据，让代码更可预测、更易测试。

## 🎯 核心概念

### 纯函数

相同输入永远返回相同输出，且没有副作用。

```javascript
// ✅ 纯函数
function add(a, b) {
  return a + b;
}

function formatName(user) {
  return `${user.firstName} ${user.lastName}`;
}

// ❌ 非纯函数（依赖外部状态）
let count = 0;
function increment() {
  count++; // 修改外部变量
  return count;
}

// ❌ 非纯函数（有副作用）
function saveUser(user) {
  database.save(user); // IO 操作
  console.log("Saved"); // 控制台输出
}
```

### 不可变性

不直接修改数据，而是创建新的数据。

```javascript
// ❌ 可变操作
const arr = [1, 2, 3];
arr.push(4); // 修改原数组

// ✅ 不可变操作
const arr = [1, 2, 3];
const newArr = [...arr, 4]; // 创建新数组

// ❌ 可变对象
const user = { name: "Alice" };
user.age = 25; // 修改原对象

// ✅ 不可变对象
const user = { name: "Alice" };
const newUser = { ...user, age: 25 }; // 创建新对象
```

## 🔧 高阶函数

接收函数作为参数或返回函数的函数。

### 函数作为参数

```javascript
// 内置高阶函数
const numbers = [1, 2, 3, 4, 5];

numbers.map((n) => n * 2); // [2, 4, 6, 8, 10]
numbers.filter((n) => n > 2); // [3, 4, 5]
numbers.reduce((sum, n) => sum + n, 0); // 15

// 自定义高阶函数
function repeat(times, fn) {
  for (let i = 0; i < times; i++) {
    fn(i);
  }
}

repeat(3, (i) => console.log(`第 ${i + 1} 次`));
```

### 函数返回函数

```javascript
// 创建乘法器
function multiplier(factor) {
  return (number) => number * factor;
}

const double = multiplier(2);
const triple = multiplier(3);

double(5); // 10
triple(5); // 15
```

## ⚡ 函数组合

将多个简单函数组合成复杂功能。

```javascript
// 基础函数
const add10 = (x) => x + 10;
const multiply2 = (x) => x * 2;
const subtract5 = (x) => x - 5;

// 手动组合
const result = subtract5(multiply2(add10(5))); // 25

// compose 函数（从右到左）
const compose =
  (...fns) =>
  (x) =>
    fns.reduceRight((acc, fn) => fn(acc), x);

const calculate = compose(subtract5, multiply2, add10);
calculate(5); // 25

// pipe 函数（从左到右，更直观）
const pipe =
  (...fns) =>
  (x) =>
    fns.reduce((acc, fn) => fn(acc), x);

const calculate2 = pipe(add10, multiply2, subtract5);
calculate2(5); // 25
```

## 🎯 柯里化

将多参数函数转换为一系列单参数函数。

```javascript
// 普通函数
function add(a, b, c) {
  return a + b + c;
}
add(1, 2, 3); // 6

// 柯里化版本
function curryAdd(a) {
  return function (b) {
    return function (c) {
      return a + b + c;
    };
  };
}
curryAdd(1)(2)(3); // 6

// 箭头函数简写
const curryAdd = (a) => (b) => (c) => a + b + c;

// 部分应用
const add1 = curryAdd(1);
const add1And2 = add1(2);
add1And2(3); // 6
```

### 通用柯里化函数

```javascript
function curry(fn) {
  return function curried(...args) {
    if (args.length >= fn.length) {
      return fn.apply(this, args);
    }
    return function (...moreArgs) {
      return curried.apply(this, args.concat(moreArgs));
    };
  };
}

// 使用
const add = (a, b, c) => a + b + c;
const curriedAdd = curry(add);

curriedAdd(1)(2)(3); // 6
curriedAdd(1, 2)(3); // 6
curriedAdd(1)(2, 3); // 6
```

## 📦 实用工具函数

### 防抖和节流

```javascript
// 防抖 - 停止触发后执行
const debounce = (fn, delay) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
};

// 节流 - 固定频率执行
const throttle = (fn, limit) => {
  let lastTime = 0;
  return (...args) => {
    const now = Date.now();
    if (now - lastTime >= limit) {
      lastTime = now;
      fn(...args);
    }
  };
};
```

### 记忆化

缓存函数结果，避免重复计算。

```javascript
const memoize = (fn) => {
  const cache = new Map();
  return (...args) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
};

// 使用
const expensiveCalc = memoize((n) => {
  console.log("计算中...");
  return n * n;
});

expensiveCalc(5); // 计算中... 25
expensiveCalc(5); // 25（直接从缓存返回）
```

## 🎮 实际应用

### 数据处理管道

```javascript
const users = [
  { name: "Alice", age: 25, active: true },
  { name: "Bob", age: 30, active: false },
  { name: "Charlie", age: 35, active: true },
];

// 函数式处理
const result = users
  .filter((u) => u.active)
  .map((u) => u.name)
  .sort();

// ["Alice", "Charlie"]
```

### React 中的函数式思想

```jsx
// 纯组件
const UserCard = ({ name, age }) => (
  <div className="card">
    <h2>{name}</h2>
    <p>Age: {age}</p>
  </div>
);

// 高阶组件
const withLoading = (Component) => (props) => {
  if (props.isLoading) return <div>Loading...</div>;
  return <Component {...props} />;
};

const UserCardWithLoading = withLoading(UserCard);
```

## 💡 最佳实践

1. **优先使用纯函数** - 易于测试和调试
2. **避免副作用** - 将副作用集中处理
3. **使用不可变数据** - 避免意外修改
4. **组合小函数** - 构建复杂功能
5. **善用 Array 方法** - map、filter、reduce

## 🔗 相关资源

- [JavaScript 基础语法](/docs/frontend/javascript/fundamentals)
- [深浅拷贝](/docs/frontend/javascript/copy)
- [ES6+](/docs/frontend/javascript/es6)

---

**下一步**：学习 [CSS 新特性](/docs/frontend/css/modern-css) 了解现代 CSS 能力。
