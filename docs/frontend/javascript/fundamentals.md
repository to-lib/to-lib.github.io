---
sidebar_position: 2
title: 基础语法
---

# JavaScript 基础语法

> [!TIP]
> 掌握函数、对象、数组等核心语法是编程的基础。

## 🎯 函数

### 函数声明

```javascript
// 函数声明
function greet(name) {
  return `Hello, ${name}!`;
}

// 函数表达式
const greet = function (name) {
  return `Hello, ${name}!`;
};

// 箭头函数
const greet = (name) => `Hello, ${name}!`;

// 调用
greet("Alice"); // "Hello, Alice!"
```

### 参数

```javascript
// 默认参数
function greet(name = "Guest") {
  return `Hello, ${name}!`;
}

// 剩余参数
function sum(...numbers) {
  return numbers.reduce((a, b) => a + b, 0);
}
sum(1, 2, 3, 4); // 10

// 解构参数
function printUser({ name, age }) {
  console.log(`${name}, ${age}岁`);
}
printUser({ name: "Alice", age: 25 });
```

### 箭头函数

```javascript
// 无参数
const sayHi = () => "Hi!";

// 单参数（可省略括号）
const double = (n) => n * 2;

// 多参数
const add = (a, b) => a + b;

// 多行（需要大括号和 return）
const process = (data) => {
  const result = data.map((x) => x * 2);
  return result;
};
```

## 📦 对象

### 创建对象

```javascript
// 对象字面量
const person = {
  name: "Alice",
  age: 25,
  greet() {
    return `Hi, I'm ${this.name}`;
  },
};

// 访问属性
person.name; // 'Alice'
person["age"]; // 25
person.greet(); // "Hi, I'm Alice"

// 修改属性
person.age = 26;
person.city = "Beijing"; // 添加新属性

// 删除属性
delete person.city;
```

### 对象方法

```javascript
const obj = { a: 1, b: 2, c: 3 };

// 获取键/值/键值对
Object.keys(obj); // ['a', 'b', 'c']
Object.values(obj); // [1, 2, 3]
Object.entries(obj); // [['a', 1], ['b', 2], ['c', 3]]

// 合并对象
const merged = { ...obj, d: 4 };
const merged2 = Object.assign({}, obj, { d: 4 });

// 检查属性
"a" in obj; // true
obj.hasOwnProperty("a"); // true
```

### 解构赋值

```javascript
const person = { name: "Alice", age: 25, city: "Beijing" };

// 基础解构
const { name, age } = person;

// 重命名
const { name: userName } = person;

// 默认值
const { country = "China" } = person;

// 嵌套解构
const user = { info: { email: "a@b.com" } };
const {
  info: { email },
} = user;
```

## 📋 数组

### 创建数组

```javascript
const arr = [1, 2, 3, 4, 5];
const arr2 = new Array(3); // [empty × 3]
const arr3 = Array.from("hello"); // ['h', 'e', 'l', 'l', 'o']
```

### 访问元素

```javascript
arr[0]; // 1
arr.at(-1); // 5 (最后一个)
arr.length; // 5
```

### 常用方法

```javascript
const arr = [1, 2, 3];

// 添加/删除
arr.push(4); // 末尾添加
arr.pop(); // 末尾删除
arr.unshift(0); // 开头添加
arr.shift(); // 开头删除

// 查找
arr.indexOf(2); // 1
arr.includes(2); // true
arr.find((x) => x > 1); // 2
arr.findIndex((x) => x > 1); // 1

// 截取
arr.slice(1, 3); // [2, 3] (不修改原数组)
arr.splice(1, 1); // 删除索引1的元素 (修改原数组)

// 合并
[1, 2].concat([3, 4]); // [1, 2, 3, 4]
[...arr1, ...arr2]; // 展开运算符
```

### 遍历方法

```javascript
const numbers = [1, 2, 3, 4, 5];

// forEach - 遍历
numbers.forEach((n) => console.log(n));

// map - 映射（返回新数组）
numbers.map((n) => n * 2); // [2, 4, 6, 8, 10]

// filter - 过滤
numbers.filter((n) => n > 2); // [3, 4, 5]

// reduce - 归约
numbers.reduce((sum, n) => sum + n, 0); // 15

// some/every - 判断
numbers.some((n) => n > 4); // true (至少一个满足)
numbers.every((n) => n > 0); // true (全部满足)

// sort - 排序
[3, 1, 2].sort((a, b) => a - b); // [1, 2, 3]
```

### 数组解构

```javascript
const [first, second, ...rest] = [1, 2, 3, 4, 5];
// first = 1, second = 2, rest = [3, 4, 5]

// 交换变量
let a = 1,
  b = 2;
[a, b] = [b, a];
```

## 🔄 条件和循环

### 条件语句

```javascript
// if-else
if (score >= 90) {
  grade = "A";
} else if (score >= 80) {
  grade = "B";
} else {
  grade = "C";
}

// 三元运算符
const result = score >= 60 ? "及格" : "不及格";

// switch
switch (day) {
  case 0:
    console.log("周日");
    break;
  case 6:
    console.log("周六");
    break;
  default:
    console.log("工作日");
}
```

### 循环语句

```javascript
// for
for (let i = 0; i < 5; i++) {
  console.log(i);
}

// for...of (遍历值)
for (const item of array) {
  console.log(item);
}

// for...in (遍历键)
for (const key in object) {
  console.log(key, object[key]);
}

// while
while (condition) {
  // ...
}

// do...while
do {
  // ...
} while (condition);
```

## 🔧 类

```javascript
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }

  greet() {
    return `Hi, I'm ${this.name}`;
  }

  static create(name) {
    return new Person(name, 0);
  }
}

// 继承
class Student extends Person {
  constructor(name, age, grade) {
    super(name, age);
    this.grade = grade;
  }

  study() {
    return `${this.name} is studying`;
  }
}

const student = new Student("Alice", 18, "A");
student.greet(); // "Hi, I'm Alice"
student.study(); // "Alice is studying"
```

## 🔗 相关资源

- [JavaScript 入门](/docs/frontend/javascript/)
- [DOM 操作](/docs/frontend/javascript/dom)
- [异步编程](/docs/frontend/javascript/async)

---

**下一步**：学习 [DOM 操作](/docs/frontend/javascript/dom) 操作网页元素。
