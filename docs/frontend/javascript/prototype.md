---
sidebar_position: 7
title: 原型链与继承
---

# 原型链与继承

> [!TIP]
> JavaScript 使用原型链实现继承，理解原型是掌握 JS 面向对象的关键。

## 🎯 原型基础

### prototype 与 **proto**

```javascript
function Person(name) {
  this.name = name;
}

Person.prototype.sayHello = function () {
  console.log(`Hello, I'm ${this.name}`);
};

const alice = new Person("Alice");

// 关系图解
// alice.__proto__ === Person.prototype
// Person.prototype.__proto__ === Object.prototype
// Object.prototype.__proto__ === null
```

### 关系说明

| 属性          | 说明                               |
| ------------- | ---------------------------------- |
| `prototype`   | 函数特有，指向原型对象             |
| `__proto__`   | 对象特有，指向构造函数的 prototype |
| `constructor` | 原型对象指回构造函数               |

```javascript
console.log(alice.__proto__ === Person.prototype); // true
console.log(Person.prototype.constructor === Person); // true
```

## 🔗 原型链

当访问对象属性时，会沿着原型链向上查找：

```javascript
const alice = new Person("Alice");

alice.sayHello(); // 在 Person.prototype 上找到
alice.toString(); // 在 Object.prototype 上找到
alice.foo; // 查到 null，返回 undefined
```

```
alice
  ↓ __proto__
Person.prototype (sayHello)
  ↓ __proto__
Object.prototype (toString, hasOwnProperty...)
  ↓ __proto__
null
```

## 📦 创建对象的方式

### 1. 构造函数

```javascript
function Animal(name) {
  this.name = name;
}

Animal.prototype.speak = function () {
  console.log(`${this.name} makes a sound`);
};

const dog = new Animal("Dog");
```

### 2. Object.create()

```javascript
const personProto = {
  greet() {
    console.log(`Hi, I'm ${this.name}`);
  },
};

const bob = Object.create(personProto);
bob.name = "Bob";
bob.greet(); // "Hi, I'm Bob"
```

### 3. ES6 Class

```javascript
class Animal {
  constructor(name) {
    this.name = name;
  }

  speak() {
    console.log(`${this.name} makes a sound`);
  }
}

const cat = new Animal("Cat");
```

## 🏗️ 继承模式

### ES6 Class 继承（推荐）

```javascript
class Animal {
  constructor(name) {
    this.name = name;
  }

  speak() {
    console.log(`${this.name} makes a sound`);
  }
}

class Dog extends Animal {
  constructor(name, breed) {
    super(name); // 必须先调用 super
    this.breed = breed;
  }

  speak() {
    console.log(`${this.name} barks`);
  }

  fetch() {
    console.log(`${this.name} fetches the ball`);
  }
}

const buddy = new Dog("Buddy", "Golden");
buddy.speak(); // "Buddy barks"
buddy.fetch(); // "Buddy fetches the ball"
```

### 寄生组合继承（ES5）

```javascript
function Animal(name) {
  this.name = name;
}

Animal.prototype.speak = function () {
  console.log(this.name + " makes a sound");
};

function Dog(name, breed) {
  Animal.call(this, name); // 继承属性
  this.breed = breed;
}

// 继承原型
Dog.prototype = Object.create(Animal.prototype);
Dog.prototype.constructor = Dog;

Dog.prototype.bark = function () {
  console.log(this.name + " barks");
};
```

## 🔍 常用方法

### 检查原型关系

```javascript
const dog = new Dog("Buddy", "Golden");

// instanceof - 检查原型链
console.log(dog instanceof Dog); // true
console.log(dog instanceof Animal); // true
console.log(dog instanceof Object); // true

// isPrototypeOf - 检查是否在原型链上
console.log(Animal.prototype.isPrototypeOf(dog)); // true

// getPrototypeOf - 获取原型
console.log(Object.getPrototypeOf(dog) === Dog.prototype); // true
```

### 属性检查

```javascript
const dog = new Dog("Buddy", "Golden");

// hasOwnProperty - 检查自有属性
console.log(dog.hasOwnProperty("name")); // true
console.log(dog.hasOwnProperty("speak")); // false

// in - 检查自有 + 原型链
console.log("name" in dog); // true
console.log("speak" in dog); // true
```

## 💡 最佳实践

### 1. 优先使用 ES6 Class

```javascript
// ✅ 推荐
class User {
  constructor(name) {
    this.name = name;
  }
}

// ❌ 避免（除非必要）
function User(name) {
  this.name = name;
}
```

### 2. 使用 super 调用父类方法

```javascript
class Dog extends Animal {
  speak() {
    super.speak(); // 先调用父类方法
    console.log("Woof!");
  }
}
```

### 3. 静态方法和属性

```javascript
class MathUtils {
  static PI = 3.14159;

  static add(a, b) {
    return a + b;
  }
}

console.log(MathUtils.PI); // 3.14159
console.log(MathUtils.add(2, 3)); // 5
```

## 🔗 相关资源

- [闭包与作用域](/docs/frontend/javascript/closure)
- [ES6+](/docs/frontend/javascript/es6)
- [基础语法](/docs/frontend/javascript/fundamentals)

---

**下一步**：学习 [错误处理](/docs/frontend/javascript/error-handling) 编写健壮的代码。
